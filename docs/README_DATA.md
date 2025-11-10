# How to Add a New Table to a Dataset in Google Cloud BigQuery

## Step 1: Navigate to BigQuery in Google Cloud Console

1. Go to the [BigQuery Console](https://console.cloud.google.com/bigquery).
2. In the **Explorer** panel, select the **project** where you want to upload the table.
3. Click on the **dataset** (e.g., `illinois_2025_nw`).
4. Click the **Create Table** button.

---

## Step 2: Configure the Table Source

1. **Create table from:** Select **Upload**.
2. **Select file:** Click **Browse**, and upload the CSV file (e.g., `iar_math.csv`).
3. **File format:** Select **CSV**.

---

## Step 3: Configure the Table Destination

1. **Project:** Choose the appropriate Google Cloud project (e.g., `nw-ai-lab`).
2. **Dataset:** Select the dataset where the table will be stored (e.g., `illinois_2025_nw`).
3. **Table name:** Enter the table name (e.g., `iar_math`).
4. **Table type:** Keep it as **Native table**.

---

## Step 4: Define the Schema

1. Under **Schema**, select **Edit as text**.
2. Copy and paste the **JSON schema** corresponding to the table (e.g., `iar_math_schema.json`).
3. Ensure that the field names and data types match the CSV file structure.

---

## Step 5: Set Advanced Options

1. Click **Advanced Options** to expand additional settings.
2. **Write preference:** Set to `Write if empty` to ensure no duplicates.
3. **Number of errors allowed:** Set to `0` to prevent unexpected data issues.
4. **Field delimiter:** Keep as **Comma (",")**.
5. **Quote character:** Select **Double quote (")**.
6. **Header rows to skip:** Set to `1` (to ignore column headers from the CSV file).
7. **Quoted newlines & Jagged rows:** Leave **unchecked** unless your data has complex formatting.

---

## Step 6: Upload and Verify

1. Click **Create Table**.
2. Wait for the upload to complete.
3. Once done, go to the dataset and click on the table.
4. Click **Preview** to ensure the data is correctly imported.
5. Run a sample query to validate:

```sql
SELECT * FROM `nw-ai-lab.illinois_2025_nw.iar_math` LIMIT 10;
```

---

## Step 7: Repeat for Additional Tables

- Repeat the process for other tables (e.g., `iar_ela.csv`).
- Use the appropriate JSON schema (e.g., `iar_ela_schema.json`).
