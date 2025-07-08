from parsing.parsers.mistral_ocr import MistralOCR

pdf_path = "F://RAG//backend//data//raw_document_data//mistral7b.pdf"
ocr = MistralOCR()
json_output = ocr.request_ocr_model(pdf_path)
print("Json_output: ", json_output)
ocr.save_json(json_output, 'F://RAG//backend//data//raw_document_data//output.json')
print("Done with saving the output....")