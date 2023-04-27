import requests
from bs4 import BeautifulSoup as bs

URL = "https://www.serebii.net/platinum/nationaldex.shtml"

def main():
    print(URL)
    page = requests.get(URL)
    soup = bs(page.content, "html.parser")
    results = soup.find_all("table", class_="dextable")
    for table in results:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            index = 0
            for cell in cells:
                #if cell.text.strip() == "" and index ==3:
                text = cell.find_all("a")
                if len(text) <= 1 and index==3:
                    for c in cells:
                        print(c.text.strip())
                index+=1
    

if __name__ == "__main__":
    main()
