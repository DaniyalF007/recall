from pypdf import PdfReader

reader = PdfReader("FinalProjectTemplates.pdf")   # put your real PDF name here

text = ""
for page in reader.pages:
    text = text + page.extract_text()

print(text)
