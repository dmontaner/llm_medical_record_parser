# colonoscopy_reporter.py
# 2023-12-15 david.montaner@gmail.com
# using LLMs to take decisions on approval of colonoscopy intervention

import os
import sys
import json
import datetime
import logging

import fitz
import torch
import chromadb

import pandas as pd

from dotenv import find_dotenv, load_dotenv

from transformers import pipeline

from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number


# Load the 'OPENAI_API_KEY' environment variable
# dotenv will search for an .env file in the parent directory structure
load_dotenv(find_dotenv())

# set simple LOGGING
if "__file__" in globals():
    log_file = os.path.basename(__file__)[:-3] + '.log'
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
else:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# ZERO SHOT MODEL used for evaluation of conservative treatments
device = 0 if torch.cuda.is_available() else -1
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
zero_shot_classes = ["Significant improvement of symptoms", "Partial improvement of symptoms", "Stable symptoms", "Worsening condition of symptoms"]


# EMBEDDINGS used to compare symptoms
embedding_function = SentenceTransformerEmbeddings(model_name="kamalkraj/BioSimCSE-BioLinkBERT-BASE")

colorectal_cancer_symptoms = [
    "changes in bowel habits",
    "persistent diarrhea or constipation",
    "changes in the size, shape, or consistency of stool",
    "narrow or ribbon-like stools",

    "blood in the stool",
    "bright red or very dark blood in the stool",
    "blood on or in the toilet tissue",

    "abdominal discomfort",
    "persistent abdominal discomfort, such as cramps, gas, or pain",
    "a feeling of fullness or bloating",

    "Cramping or abdominal (belly) pain",

    "unexplained weight loss",
    "significant and unexplained weight loss over a short period",

    "fatigue",
    "persistent fatigue or weakness",

    "iron deficiency anemia",
    "a low red blood cell count due to chronic bleeding, which can lead to anemia",

    "incomplete evacuation",
    "the feeling of incomplete bowel movement after going to the bathroom",

    "unexplained anorexia",
    "loss of appetite without an obvious cause",
]

jps_symptoms = [
    "gastrointestinal bleeding",
    "recurrent bleeding from the gastrointestinal tract",
    "blood in the stool",
    "black, tarry stools (melena)",

    "anemia",
    "iron deficiency anemia, characterized by fatigue, weakness, and pale skin",

    "abdominal pain",
    "abdominal pain or discomfort",

    "diarrhea",
    "inflammation associated with the polyps",

    "intussusception",
    "a segment of the intestine folds into itself, causing a blockage",
    "abdominal pain, vomiting, and bloody stools",
]

_documents = colorectal_cancer_symptoms + jps_symptoms
_metadatas = [{"source": "colorectal_cancer_symptoms"} for _ in colorectal_cancer_symptoms] + [{"source": "jps_symptoms"} for _ in jps_symptoms]
_embeddings = embedding_function.embed_documents(_documents)
_ids = [str(i) for i in range(len(_documents))]

chroma_client = chromadb.Client()

chroma_client.clear_system_cache()
if "symptoms" in chroma_client.list_collections():
    chroma_client.delete_collection("symptoms")

chroma_collection = chroma_client.create_collection(name="symptoms", metadata={"hnsw:space": "cosine"})
chroma_collection.add(
    documents=_documents,
    metadatas=_metadatas,
    embeddings=_embeddings,
    ids=_ids,
)


def is_a_symptom(s, source, cutoff=0.2):
    if not isinstance(s, list):
        s = [s]
    res = chroma_collection.query(
        query_embeddings=embedding_function.embed_documents(s),
        n_results=1,
        where={"source": source})

    dist = [x[0] for x in res["distances"]]
    is_symptom = [x <= cutoff for x in dist]
    return is_symptom


# KOR EXTRACTOR
kor_schema = Object(
    id="patient_info",
    description="Relevant personal and medical information extracted from medical records.",

    examples=[
        ("""
        Name: David Montaner
        Procedure: 32454
        DOB: 07/03/1977
        """,
         [
             {
                 "name": "David Montaner",
                 "requested_procedure": "32454",
                 "dob": "07/03/1977",
             }
         ]
         ),

        ("""
        Name: J. Peterson
        Requested intervention: 19782
        Born in 04/14/1988
        ...  diet changes did not improve the patient condition but symptoms completely remitted with paracetamol treatment ...
        """,
         [
             {
                 "name": "J. Peterson",
                 "requested_procedure": "19782",
                 "dob": "04/14/1988",
                 "conservative_treatment": ["diet changes did not improve the patient condition", "symptoms completely remitted with paracetamol treatment"]
             }
         ]
         ),
    ],

    attributes=[
        Text(
            id="name",
            many=False,
            description="The name of the patient.",
        ),
        Text(
            id="dob",
            many=False,
            description="The date of birth of the patient.",
        ),
        Number(
            id="age",
            many=False,
            description="The age of the patient.",
        ),
        Text(
            id="requested_procedure",
            many=True,
            description="The procedure or procedures finally recommended or indicated for the patient. Generally given as Current Procedural Terminology (CPT) codes which are five-digit numeric codes; sometimes they can be followed by two additional digits for further specificity.",
        ),
        Text(
            id="intervention_planned_date",
            many=True,
            description="The day for which the intervention is planned.",
        ),
        Text(
            id="previous_colonoscopy_date",
            description="The date in which any previous colonoscopy was carried out.",
            many=True,
        ),
        Text(
            id="current_symptoms",
            many=True,
            description="Current symptoms, signs or conditions which have led to the doctor to propose the intervention.",
        ),
        Text(
            id="family_colorectal_cancer",
            many=True,
            description="Colorectal cancer diagnosed in one or more first-degree relatives.",
        ),
        Text(
            id="family_adenomatous_polyposis",
            many=True,
            description="Colonic adenomatous diagnosed in an family member.",
        ),
        Text(
            id="conservative_treatment",
            many=True,
            description="Identify which conservative treatments, if any, have been successfully applied prior to the Colonoscopy request. By conservative treatment we refer to a non-invasive or non-surgical approach to managing the symptoms or patient discomfort.",  # Please return the whole sentence including the positive result of the treatment.",
        ),
    ],
    many=False
)

llm = ChatOpenAI(
    model="gpt-4-0613",
    temperature=0,
)

chain = create_extraction_chain(llm, kor_schema, encoder_or_encoder_class="json")


# FINAL DECISION TEMPLATE
decision_template = """
ACCEPTED COLONOSCOPY: {accept_colonoscopy}

REASON:
{reason}

NOTES:
{notes}
"""


# MAIN CLASS and pipeline
class ColonoscopyRecord:
    """
    Automatizes the guidelines check for colonoscopy medical records.
    """

    def __init__(
            self,
            pdf_filename=None,
            pdf_stream=None,
            conservative_treatments_cutoff=0.7,
            symptoms_cutoff=0.2,
            log=None,
    ):
        self.pdf_filename = pdf_filename
        self._pdf_stream = pdf_stream
        self.conservative_treatments_cutoff = conservative_treatments_cutoff
        self.symptoms_cutoff = symptoms_cutoff
        self.log = log if log else logging
        self.text = None
        self._main_requested_procedure = "45387"

    def read_pdf(self):
        try:
            if self._pdf_stream:
                doc = fitz.open(stream=self.pdf_stream)
            else:
                doc = fitz.open(filename=self.pdf_filename)

            self.text = ""
            for page in doc:
                self.text += page.get_text()

            doc.close()
            self.log.debug("PDF read by PyMuPDF fitz")

        except Exception as e:
            self.log.error("PyMuPDF fitz could not read the PDF")
            self.log.error(e)

        if not self.text:
            try:
                loader = PyPDFLoader(self.pdf_filename)
                pages = loader.load()
                self.text = "\n".join([x.page_content for x in pages])
                self.log.debug("PDF read by langchain PyPDFLoader")
            except Exception as e:
                self.log.error("langchain PyPDFLoader could not read the PDF")
                self.log.error(e)

    def extract_information(self):
        try:
            res = chain.run(self.text)
            self.extracted_info = res["data"]["patient_info"]
            if isinstance(self.extracted_info, list):  # because sometimes the result is wrapped in a list
                self.extracted_info = self.extracted_info[0]
            self.log.debug("information extracted via KOR chain")
        except Exception as e:
            self.log.error("KOR chain could not extract information")
            self.log.error(e)

    def find_conservative_treatments(self, cutoff=None):
        treatments = self.extracted_info.get("conservative_treatment", [])
        if treatments:
            cutoff = cutoff if cutoff else self.conservative_treatments_cutoff
            df = []
            for treatment in treatments:
                res = zero_shot_pipeline(treatment, candidate_labels=zero_shot_classes)
                df.append(pd.DataFrame(res))
            df = pd.concat(df).reset_index(drop=True)
            df["successfully"] = (cutoff < df["scores"]) & (df["labels"] == "Significant improvement of symptoms")
            self._conservative_treatments_df = df
        else:
            self._conservative_treatments_df = None

    @property
    def conservative_treatments(self):
        if self._conservative_treatments_df is None:
            return []
        else:
            return self._conservative_treatments_df.query("successfully")["sequence"].tolist()

    @property
    def had_successful_conservative_treatments(self):
        return 0 < len(self.conservative_treatments)

    @property
    def _current_date(self):
        try:
            return pd.to_datetime(self.extracted_info.get("intervention_planned_dat", [datetime.datetime.now()])).min()
        except Exception:
            return None

    @property
    def age(self):
        age = None
        try:
            d1 = self._current_date
            d0 = pd.to_datetime(self.extracted_info["dob"])
            age = int((d1 - d0).days / 365)
        except Exception as e:
            self.log.debug("could not compute age from DOB")
            self.log.debug(e)

        if age is None:
            try:
                age = int(self.extracted_info.get("age"))
            except Exception as e:
                self.log.debug("age was not extracted")
                self.log.debug(e)

        return age

    def had_previous_colonoscopy(self, years=10):
        previous_colonoscopy_date = self.extracted_info.get("previous_colonoscopy_date")
        if previous_colonoscopy_date:
            latest_colonoscopy_date = pd.to_datetime(previous_colonoscopy_date).max()
            return (self._current_date - latest_colonoscopy_date).days < 3650
        else:
            return False

    def is_symptomatic(self, source, cutoff=None):
        cutoff = cutoff if cutoff else self.symptoms_cutoff
        symptoms = self.extracted_info.get("current_symptoms", [])
        if symptoms:
            res = is_a_symptom(symptoms, source, cutoff=0.2)
            return any(res)
        else:
            return False

    def guidelines_patient_at_risk(self):
        """
        Patient has average-risk or higher, as indicated by ALL of the following
        - Age 45 years or older
        - No colonoscopy in past 10 years
        """
        return self.had_previous_colonoscopy() & (45 <= self.age)

    def guidelines_family_risk(self):
        """
        High risk family history, as indicated by 1 or more of the following:
        - Colorectal cancer diagnosed in one or more first-degree relatives of any age and ALL of the following:
          + Age 40 years or older
          + Symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding)
        - Family member with colonic adenomatous polyposis of unknown etiology
        """
        family_colorectal_cancer = 0 < len(self.extracted_info.get("family_colorectal_cancer", []))
        age40 = 40 <= self.age
        symptomatic = self.is_symptomatic(source="colorectal_cancer_symptoms")

        return family_colorectal_cancer & age40 & symptomatic

    def guidelines_polyposis(self):
        """
        Juvenile polyposis syndrome diagnosis indicated by 1 or more of the following:
        - Age 12 years or older     and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia)
        - Age younger than 12 years and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia)
        """
        symptomatic = self.is_symptomatic(source="jps_symptoms")

        return symptomatic

    def main_pipeline(self):
        """
        This method drives the main logic to accept or reject the request.
        Notice that if the request is not for a colonoscopy
        """
        self.read_pdf()
        self.extract_information()
        self.find_conservative_treatments()

        # check if the requested procedures are for a colonoscopy
        req_proc = self.extracted_info.get("requested_procedure")
        if self._main_requested_procedure in req_proc:
            self.notes = None
        else:
            self.notes = f"Notice that the requested procedures {req_proc} are not for colonoscopy ({self._main_requested_procedure})"

        # Identify if conservative treatment prior to the Colonoscopy request has already been attempted
        # a. If conservative treatment was successful, do not continue through the pipeline and instead
        #    present evidence that conservative treatment improved the patient’s condition
        if self.had_successful_conservative_treatments:
            self.accept_colonoscopy = False
            self.reason = '\n'.join(
                ['Previous conservative treatment was successful as indicated in the report:'] + self.conservative_treatments
            )
        # b. If conservative treatment has failed, continue through the pipeline and satisfy the following criteria
        else:
            if self.guidelines_patient_at_risk():
                self.accept_colonoscopy = True
                self.reason = (
                    "Patient has average-risk or higher, as indicated by ALL of the following\n"
                    "- Age 45 years or older\n"
                    "- No colonoscopy in past 10 years\n"
                )
            elif self.guidelines_family_risk():
                self.accept_colonoscopy = True
                self.reason = (
                    "High risk family history, as indicated by 1 or more of the following:\n"
                    "- Colorectal cancer diagnosed in one or more first-degree relatives of any age and ALL of the following:\n"
                    "+ Age 40 years or older\n"
                    "+ Symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding)\n"
                    "- Family member with colonic adenomatous polyposis of unknown etiology\n"
                )
            elif self.guidelines_polyposis():
                self.accept_colonoscopy = True
                self.reason = (
                    "Juvenile polyposis syndrome diagnosis indicated by 1 or more of the following:\n"
                    "- Age 12 years or older     and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia)\n"
                    "- Age younger than 12 years and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia)\n"
                )
            else:
                self.accept_colonoscopy = False
                self.reason = "We did not find evidence in the document that supported any of the colonoscopy guidelines. Please provide further information for this procedure to be approved."

    def format_final_decision(self):
        res = decision_template.format(
            accept_colonoscopy=self.accept_colonoscopy,
            reason=self.reason,
            notes=self.notes,
        )
        return res

    def report(self, text=False):
        res = {}
        res["filename"] = self.pdf_filename
        res["name"] = self.extracted_info.get("name")
        res["dob"] = self.extracted_info.get("dob")
        res["age"] = self.age
        res["requested_procedure"] = self.extracted_info.get("requested_procedure")
        res["intervention_planned_date"] = self.extracted_info.get("intervention_planned_date")
        res["current_symptoms"] = self.extracted_info.get("current_symptoms")
        res["family_colorectal_cancer"] = self.extracted_info.get("family_colorectal_cancer")
        res["conservative_treatments"] = self.conservative_treatments
        res["had_previous_colonoscopy"] = self.had_previous_colonoscopy()

        res["risk_patient"] = self.guidelines_patient_at_risk()
        res["risk_family"] = self.guidelines_family_risk()
        res["risk_polyposis"] = self.guidelines_polyposis()

        if text:
            res["text"] = self.text
        return res

    def print_report_to_json(self, text=False):
        res = self.report(text=text)
        print(json.dumps(res, indent=2))


# COMMAND LINE TOOL
if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Determine if the medical record satisfies the guidelines for Colonoscopy"
    )
    parser.add_argument("filepath", type=str, help="path to the PDF file containing the report")
    args = parser.parse_args()

    print("\nMAIN PIPELINE STARTED\n", flush=True)
    record = ColonoscopyRecord(pdf_filename=args.filepath)
    record.main_pipeline()
    print(record.format_final_decision())
    record.print_report_to_json()
