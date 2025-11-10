# Setting Up an SQL BI Agent in Google Cloud

## Step 1: Prepare BigQuery Dataset and Schema

See **[How to Add a New Table to a Dataset in Google Cloud BigQuery](../datasets/README.md)**

## Step 2: Set Up Vertex AI Agent Builder

1. Navigate to **Vertex AI → Agent Builder** in Google Cloud.

2. Create a new **Data Store**:

  * Select **BigQuery as Source**.

  * Choose **Structured Data**.

  * Point it to your **BigQuery dataset (`tables_descriptions`)**.

  * Ensure the `Retrievable` checkbox of Field name `description` is checked.

  * Set **global location** and finalize setup.

3. Create an **Application**:

  * Go to **Vertex AI** → **Agent Builder** → **Apps**.

  * Click **Create New App**.

  * Choose **Search for your website** option.

  * Name it `table_search_app`.

  * Link it to the previously created **Data Store**.

4. Copy **Data Store ID and Region** for later use.

## Step 3: Run the SQL BI Agent locally

1. **Service account**: In the Google Cloud Console, go to IAM & Admin > Service Accounts and create a new service account. Contact the Google Cloud Project Administrator (Fernando) in case you don't have permissions to view service accounts.

2. **Grant necessary permissions**: Grant the service account the necessary permissions to access the resources your application needs, such as Vertex AI and BigQuery

3. Download service account key: Download the service account key file (JSON format) to your local machine.

4. Set environment variable: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the downloaded service account key file. You can do this in your terminal or command prompt using:

```zsh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
```

5. Run the Workflow Locally

```zsh
streamlit run streamlit.py
```

6. Input question such as "How many students scored above 700 in math?"

## Step 4: Test and Validate

* Run sample queries.

* Check logs in **Vertex AI Agent Builder**.

* Ensure the agent retrieves and processes queries correctly.
