from extractor import KTPOCR
import os

if __name__ == "__main__":
    paths = "./dataset/"
    for file in os.listdir(paths):
        ocr = KTPOCR(paths+file)
        ocr.run()