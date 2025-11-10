# Making Changes to BigQuery Tables

Throughout our development process, we found that whenever a change was made to a table, particularly in reference to descriptions of columns or tables for better indexing and search, the whole configuration process of enabling a Vertex Agent Builder app would have to be restarted, including:

1. Re-uploading changed tables to BigQuery, especially `tables_descriptions`. We are unsure whether changing a child table such as `iar_math` or `iar_ela` would require a restart of the cloud setup.

2. Re-creating a Vertex AI Agent Builder Data Store.

3. Re-creating a Vertex AI Agent Builder app.

4. Changing the `vertex_agent_builder_data_store_id` within `settings.py` to the new Vertex AI Agent Builder app's ID.
