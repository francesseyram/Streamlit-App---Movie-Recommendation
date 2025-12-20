
🎬 Movie Analytics & Recommendation Dashboard

An interactive analytics and recommendation dashboard built for a streaming technology company use case.
This application transforms large-scale movie rating data into actionable business insights and personalized discovery tools.



Project Overview

Modern streaming platforms depend on data-driven systems to understand user behavior, evaluate content performance, and improve content discovery. This Streamlit application serves as the deployment layer of a broader Movie Analytics and Recommendation System.

The dashboard enables stakeholders to:  
	•	Explore global and genre-level rating trends  
	•	Analyze user rating behavior and activity patterns  
	•	Discover high-quality but low-visibility movies (“hidden gems”)  
	•	Interactively inspect analytics derived from large-scale user–movie interaction data  

The app is powered by a Parquet dataset generated during the Exploratory Data Analysis (EDA) phase, ensuring consistency between offline analysis and online visualization.



Key Features  

 Ratings Overview  
	•	Summary statistics (mean, median, standard deviation)  
	•	Interactive rating distribution visualizations  
	•	High-level view of platform-wide user sentiment  

 Genre Analytics  
	•	Average rating and popularity by genre  
	•	Comparison between content quality and engagement  
	•	Identification of niche, high-performing genres  

 User Behavior Analysis  
	•	Segmentation of users into harsh, moderate, and generous raters  
	•	Identification of highly active users  
	•	Visualization of rating bias and engagement patterns  

 Hidden Gems Finder  
	•	Discovery of movies with high ratings but low visibility  
	•	Adjustable thresholds for rating quality and popularity  
	•	Interactive tables and scatter plots for exploration  




 Data Pipeline  
	1.	Raw MovieLens datasets are cleaned, enriched, and analyzed during the EDA phase.  
	2.	Feature-engineered outputs are exported to a Parquet file.  
	3.	The Streamlit app loads this Parquet file directly for fast, consistent analytics rendering.  

This approach ensures:  
	•	Reproducibility between analysis and deployment  
	•	Efficient data loading  
	•	Clear separation between data processing and presentation layers  


 Live Deployment  

The application is deployed and accessible at:  

🔗 https://francesseyram-frances.hf.space  



 Running the App Locally  

Prerequisites  
	•	Python 3.9+  
	•	Required libraries listed in requirements.txt  

Installation & Execution  

pip install -r requirements.txt  
streamlit run streamlit_app.py  

Ensure that the Parquet file generated from the EDA phase is available in the expected directory as referenced in streamlit_app.py.  



 Technologies Used  
	•	Python  
	•	Streamlit – Web application framework  
	•	Pandas & NumPy – Data manipulation  
	•	Plotly / Matplotlib / Seaborn – Data visualization  
	•	Parquet – Efficient columnar data storage  



 Business Value  

This dashboard supports strategic decision-making by:  
	•	Improving content discovery and catalog utilization  
	•	Highlighting underexposed high-quality movies  
	•	Informing personalization and recommendation strategies  
	•	Providing analysts and managers with real-time insight exploration tools  



Ethical Considerations  
	•	User data is anonymized and contains no personally identifiable information  
	•	Rating bias is explicitly analyzed to reduce unfair model influence  
	•	Analytics are designed to promote content diversity rather than popularity-only exposure  


Notes

This dashboard is part of a larger academic project focused on building an end-to-end movie analytics and recommendation system, including exploratory analysis, machine learning models, deployment, and business storytelling.

