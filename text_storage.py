import pyodbc
import json
from bs4 import BeautifulSoup
from text_transform import process_text

def fetch_text_from_db():
    connection = pyodbc.connect(
        r'Driver=SQL Server;Server=.\SQLEXPRESS;Database=cwdb;Trusted_Connection=yes;'
    )
    cursor = connection.cursor()

    res = []
    cursor.execute('SELECT ID, Description FROM Courses')
    for row in cursor:
        soup = BeautifulSoup(row.Description, 'html.parser')
        description = ' '.join(list(soup.stripped_strings))
        res.append({'id': row.ID, 'description': description})

    connection.close()
    return res

def fetch_courses_from_db():
    connection = pyodbc.connect(
        r'Driver=SQL Server;Server=.\SQLEXPRESS;Database=cwdb;Trusted_Connection=yes;'
    )
    cursor = connection.cursor()

    courses = []
    cursor.execute("""
        SELECT [ID]
            ,[Title]
            ,[Rating]
            ,[Hours]
            ,[Url]
            ,[Description]
            ,[PriceDetail_Amount]
            ,[PriceDetail_Currency]
            ,[PriceDetail_CurrencySymbol]
            ,[PriceDetail_PriceString]
        FROM [cwdb].[dbo].[Courses]
    """)
    for row in cursor:
        courses.append({
            "id": row.ID,
            "title": row.Title,
            "rating": row.Rating,
            "hours": row.Hours,
            "url": row.Url,
            "description": row.Description,
            "priceDetail": {
                "amount": row.PriceDetail_Amount,
                "currency": row.PriceDetail_Currency,
                "currencySymbol": row.PriceDetail_CurrencySymbol,
                "priceString": row.PriceDetail_PriceString
            }
        })

    connection.close()
    return courses

def save_to_json():
    data = fetch_text_from_db()
    text = list(map(lambda obj: obj['description'], data))
    processed_text = process_text(text)
    for i, _ in enumerate(data):
        data[i]['description'] = processed_text[i]
    with open('corpus.json', 'w', encoding='utf-8') as outfile:
        json.dump(
            data, 
            outfile, 
            indent=4, 
            sort_keys=False,
            ensure_ascii=False,
            separators=(',', ': ')
        )

def read_from_json():
    data = []
    with open('corpus.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

if __name__ == '__main__':
    save_to_json()