# Using LLMs to parse medical records from PDF files

__colonoscopy_reporter.py__ is a Python script designed to automate 
the evaluation of colonoscopy intervention requests based on medical records.
It utilizes various language models and databases to make decisions
on the approval of colonoscopy interventions.

## Set Up

Cone the repository and then:

```
virtualenv --python=python3.10 venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt 
```

### Prerequisites

You will need to set up the `OPENAI_API_KEY` environment variable as the tool makes calls to the
[OpenAI API](https://platform.openai.com/docs/api-reference?lang=python).

You can do this creating a file called `.env` with this line in it:

```
OPENAI_API_KEY=*****************************************************

```

The first time that you run the tool it will attempt to download several models from the 
[Hugging Face Models](https://huggingface.co/models)
site.
__This may take some time__ the first time.


## Usage

### Command line

Run the script from the command line, providing the path to the PDF file containing the colonoscopy report:

    python colonoscopy_reporter.py data/medical-record-1.pdf 

The script will analyze the provided medical report and generate a decision based on predefined guidelines.

### From Python

The main class in the module is `ColonoscopyRecord`.
You can use it as follows:

```
from colonoscopy_reporter import ColonoscopyRecord

file = f"data/medical-record-1.pdf"
record = ColonoscopyRecord(pdf_filename=file)
record.main_pipeline()
print(record.format_final_decision())
record.print_report_to_json()
```


# IMPLEMENTATION

## Reading PDF
The script uses the PyMuPDF (fitz) library or langchain's PyPDFLoader to read the contents of the PDF file. It attempts both methods, providing flexibility in handling different PDF formats.

## Information Extraction
The [KOR](https://eyurtsev.github.io/kor/) package (Knowledge Oriented Reasoning) extraction chain is employed to extract relevant personal and medical information from the medical records. The extracted information includes patient details, requested procedures, date of birth, and more.

## Conservative Treatment Analysis
The script analyzes conservative treatments mentioned in the report using a __zero-shot classification model__ from Hugging Face. 
If successful treatments are identified, the colonoscopy request may be rejected based on evidence of prior improvement.

## Symptoms Evaluations
Symptoms for "colorectal cancer" and for "juvenile polyposis syndrome" are stored in ChromaDB Vector Database.
Then all symptoms found in the report are compared to those in the vector database.
If they are similar enough we report positive symptoms for the corresponding condition.

Another model from Hugging Face is used to compute the embeddings.
In this case we use the model "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
which has been trained on biomedical literature.

## Guidelines Check
The script follows specific guidelines to determine if the colonoscopy request aligns with accepted medical criteria. 
Guidelines include considerations for patient age, family history, and symptoms associated with colorectal cancer or juvenile polyposis syndrome.

## Final Decision
Based on the information extracted and the guidelines check, the script makes a final decision on whether to accept or reject the colonoscopy request. If rejected, the reason is provided.

## Reporting
The script generates a detailed report containing patient information, age, requested procedure, symptoms, family history, and more. This report is printed to the console in JSON format.
