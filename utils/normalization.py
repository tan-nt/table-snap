from bs4 import BeautifulSoup

def normalize_rows_and_headers(headers, rows):
    # If there are no rows, return None, None
    if len(rows) == 0:
        return None, None

    if len(headers) == 0:
        headers = rows[0]
        rows = rows[1:]

    # Determine the maximum number of columns in the rows
    valid_row_lengths = [len(row) for row in rows if len(row) > 0]
    if not valid_row_lengths:
        max_columns = len(headers)  # Use header length if no valid rows
    else:
        max_columns = max(valid_row_lengths)

    # Ensure the headers match the maximum number of columns
    if len(headers) < max_columns:
        headers.extend([''] * (max_columns - len(headers)))

    # Normalize rows to match header length
    for row in rows:
        if len(row) != len(headers):
            row.extend([''] * (len(headers) - len(row)))  # Add empty strings to rows with fewer columns

    return headers, rows


def convert_to_html_page(headers, rows, title="Table Output"):
    html = "<!DOCTYPE html>\n"
    html += "<html lang='en'>\n"
    html += "<head>\n"
    html += f"  <meta charset='UTF-8'>\n"
    html += f"  <title>{title}</title>\n"
    html += "  <style>\n"
    html += "    table { border-collapse: collapse; width: 100%; }\n"
    html += "    th, td { border: 1px solid black; padding: 8px; text-align: left; }\n"
    html += "    th { background-color: #f2f2f2; }\n"
    html += "  </style>\n"
    html += "</head>\n"
    html += "<body>\n"
    html += f"<h2>{title}</h2>\n"
    html += "<table>\n"

    # Add header row
    html += "  <tr>\n"
    for header in headers:
        html += f"    <th>{header}</th>\n"
    html += "  </tr>\n"

    # Add data rows
    for row in rows:
        html += "  <tr>\n"
        for cell in row:
            html += f"    <td>{cell}</td>\n"
        html += "  </tr>\n"

    html += "</table>\n"
    html += "</body>\n"
    html += "</html>"
    return html


def normalize_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Step 2: Find the table in the HTML (assuming the first <table> is what you want)
    table = soup.find('table')

    # Step 3: Extract the headers (th elements)
    headers = [header.text.strip() for header in table.find_all('th')]

    # Step 4: Extract the rows (tr elements)
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        columns = row.find_all('td')
        if columns:
            rows.append([col.text.strip() for col in columns])

    headers, rows = normalize_rows_and_headers(headers, rows)
    return convert_to_html_page(headers, rows)

def html_to_csv(html):
    print('html', html)
    # Step 1: Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Step 2: Find the table in the HTML (assuming the first <table> is what you want)
    table = soup.find('table')

    # Step 3: Extract the headers (th elements)
    headers = [header.text.strip() for header in table.find_all('th')]

    # Step 4: Extract the rows (tr elements)
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        columns = row.find_all('td')
        if columns:
            rows.append([col.text.strip() for col in columns])

    headers, rows = normalize_rows_and_headers(headers, rows)

    print('headers', headers)
    print('rows', rows)
    # Step 5: Create a DataFrame from the extracted data
    df = pd.DataFrame(rows, columns=headers)

    # Step 6: Save DataFrame to CSV
    csv_filename = "table_output.csv"
    df.to_csv(csv_filename, index=False)
    return df
