# Using LLMs to parse medical records from PDF files

## Set Up

Cone the repository and then:

```
virtualenv --python=python3.10 venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt 
```

### Requirements

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

    python colonoscopy_reporter.py data/medical-record-1.pdf 


### From Python

```
from colonoscopy_reporter import ColonoscopyRecord

file = f"data/medical-record-{i}.pdf"
print('=' * 100, flush=True)
print(file)
    
record = ColonoscopyRecord(pdf_filename=file)
record.main_pipeline()
print(record.format_final_decision())
record.print_report_to_json()
```
