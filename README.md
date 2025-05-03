# Historical Document Extraction with AI

## Problem
**Task:** A collection of emails have been printed and scanned. I believe it will be useful to me to have **create a semi-structured database** of the information contained in the emails.

Here is an example of an email and what we wnat to extract from it:

![IE_example](IE_example.png)


Doing this by hand would be tedious! Let's use **AI** creatively instead. I could choose a supervised approach, in which I annotate some number of documents myself and then train models on them. This is still fairly tedious, so I designed an **unsupervised** approach in which I don't need to annotate data.

## Approach
### Overview
My approach can be boiled down to 3 steps (and 3 corresponding Colab notebooks):

1. **Optical Character Recognition (OCR)**: converting image files into text that can be processed by LLMs.
2. **Schema Inference (SI)**: prompting an LLM to generate a JSON schema capturing the commonalities among a few sample emails.
3. **Information Extraction (IE)**: prompting an LLM with constrained generation to extract information based on the inferred schema.

Essentially, images are converted to text, a schema is inferred from the text, and then information is extracted from the text according to the specified schema.

![concept_diagram](concept_diagram.png)

### 1. Optical Character Recognition (OCR)
The goal of OCR is to convert the text present in an image into textual data. I used **PaddleOCR**, which contains a pipeline of neural networks (CNNs and RNNs) that detect regions of an image containing text and then extract the text within them.

### 2. Schema Inference (SI)
The goal of SI is to automatically infer schema from the texts—perhaps just from a subset of them. This schema will eventually be used to guide the extraction of information in a semi-structured JSON object. I fed **3 sample emails** to a **LLaMa 3** LLM and instructed it to spot structural elements common to all of them and generate a corresponding JSON schema. 

### 3. Information Extraction (IE)
The goal of IE here is to automatically generate structured records, which follow the schema, from the unstructured text. I prompted a **LlaMa 3** LLM to extract the structured records from the emails, one at a time. I used **Outlines** to perform **constrained generation**, which forces the LLM to generate according to the schema.

## How good is this system?
Here we examine one email as it passes through the pipeline. Here is the raw image:

![raw](truncated_80909413.png)
### OCR
The OCR was nearly perfect, as we can see in the results below.

![raw](ocr_80909413.png)

### Schema Inference
The schema that I inferred in an automated fashion (below) is very high quality. This may have been particularly successful because the LLM is surely familiar with the format of emails. An illuminating experiment would be to try generating a schema without any sample documents.

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Common Email Structure",
  "type": "object",
  "properties": {
    "subject": {
      "type": "string"
    },
    "from": {
      "type": "string"
    },
    "to": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "cc": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "body": {
      "type": "string"
    },
    "date": {
      "type": "string"
    }
  },
  "required": [
    "subject",
    "from",
    "to",
    "cc",
    "body",
    "date"
  ]
}
```

### Information Extraction
Information extraction was of moderate quality; there were frequent errors. Many of these errors are small and understandable. Many errors could be solved with more advanced constrained generation, in which only the only text that could fill slots in a record is exact quotations from the source text. Here we mark the correct and incorrect information in the extracted record:

![IE_errors](ie_errors.png)

## Challenges & Solutions

- I was **limited by computational resources** because I mainly have access to a personal laptop and I refrained from using state-of-the-art private models available via API as I am given to understand that the security and privacy needs of the company preclude their use. Therefore, I was limited to models with 1 billion parameters, which necessitate more care in prompting, more forethought in generation methods, and more creative methods.
- I am familiar with computer vision research, I have limited hands-on experience—I previously specialized in natural language processing, so I had to learn what worked and what didn't. I originally tried to use **Tesseract** for OCR, but the results were poor. I then tried to switch to PaddleOCR, but evidently it is rarely compatible with Apple computers (what I own) despite the existence of installation instructions for them. This is why I resorted to **Colab** notebooks, where I could run Tesseract, instead of my usual VSCode + GitHub stack.
- During Schema Inference, prompts I wrote returned generations that were variously **nonsensical**, **unruly**, and **repetitive**. Unlike OCR, natural language processing is my specialty, and I was familiar with these issues as well as tricks of the trade to solve them; this was a lesser challenge. A combination of an **imperious tone** and **extreme seriousness** of certain details in the prompt as well as **deterministic generation** solved the issue.
- Vanilla use of generation with LlaMa 3 usually resulted in **invalid JSON** objects. I knew that **constrained generation** was the solution for this, and I found a library (`outlines`) that supports it.
