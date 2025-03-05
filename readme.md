Project in development:

 Aim: The Dataset is uploaded via a website which uploads to ADLS. When the button is clicked, it will trigger the pipeline to run in Azure Data Factory. ADF will run databricks to preprocess the data, then it will ingests the data into a new container in ADLS. From there Azure ML will use the processed dataset to build a deep learning model which will segregate the tumours. 

 Steps:
 1. Develop a website UI which can upload datasets into ADLS Containers.
 2. Use Databricks for preprocessing  the data uploaded and ingest into a new container.
 3. Extract the preprocessed dataset into Azure ML services, then develop deep learning models. Select which one has best accuracy to classify any new images.
 4. Ingest all these seggregated images into new container.
 5. All these orchestrated and run by Data Factory when clicked a button called "Classify" on the website developed.

Problem: We need huge database of MRI/CT scans of brain tumors.

 Techstack:
 Cloud: Azure ML(need to learn), DataBricks, Data Factory, DataLakeHouse.
 Web Development: HTML, CSS, JavaScript, React JS.
 Languages: Python (Pyspark)
 What I have to do: Learn how to use Azure ML platform and find brain tumour datasets.
 When to Complete: By April Ending.
 Who are your groupmates: Surya.G
 Whats the use: 
           1. Able to learn and use CNN's
           2. Able to integrate Azure Services with web development.
           3. This pipeline can be used in HealthCare for other purposes.
           4. As per the Course Requirements, we can simply develop  deep learning models but integrating all these services will hone our skills.
Under Guidance of Dr. Hembroff

