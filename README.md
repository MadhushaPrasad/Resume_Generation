# Resume_Generation

Here's a summary of what the code does:

- It downloads necessary NLTK resources for text processing.
- It defines labeled data consisting of resumes and their corresponding IT industry labels.
- It preprocesses the labeled data by tokenizing, converting to lowercase, and removing stopwords.
- It splits the preprocessed data into training and testing sets.
- It constructs a pipeline that includes TF-IDF vectorization and logistic regression model.
- It trains the model on the training data.
- It evaluates the model's accuracy using the testing data and prints the accuracy score.
- It visualizes the confusion matrix to assess the model's performance.
- It extracts relevant text data from the provided resume JSON object.
- It preprocesses the extracted text data.
- It uses the trained model to predict the suitable IT industry for the resume.
