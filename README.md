# MailGuard AI: Spam Email Classifier

MailGuard AI is a Machine Learning-based application designed to classify emails as spam or not spam, helping users safeguard their inboxes from unwanted or malicious content. Built with Streamlit for an interactive and user-friendly interface, MailGuard AI leverages a pre-trained model to deliver accurate results.

---

## Features

- **Spam Detection**: Classifies email content into "Spam" or "Not Spam."
- **Confidence Score**: Displays the model's confidence level in the classification.
- **Interactive UI**: A modern, intuitive interface using Streamlit.
- **AI-Powered**: Utilizes Machine Learning for fast and reliable predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sahilganjave05/MailGuardAI.git
   ```

2. Navigate to the project directory:
   ```bash
   cd MailGuardAI
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the `spam.pkl` (model) and `vec.pkl` (vectorizer) files in the project directory.

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Enter the email content in the text area and click "Classify Email" to see the results.

---

## Project Structure

- `app.py`: Main application file.
- `spam.pkl`: Pre-trained model for spam classification.
- `vec.pkl`: Vectorizer for text data preprocessing.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

---

## Example Output

1. **Not Spam**:
   - Input: "Your order has been shipped and will arrive soon."
   - Output: "✅ This is Not a Spam Email (95.67% confidence)."

2. **Spam**:
   - Input: "Congratulations! You've won a $1000 gift card. Click here to claim."
   - Output: "🚫 This is a Spam Email (89.23% confidence)."

---

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Web application framework.
- **scikit-learn**: For model and vectorizer.
- **Pickle**: Model serialization.

---

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and push to your fork.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any inquiries or feedback, please reach out to:
- **Sahil Ganjave**
- GitHub: [sahilganjave05](https://github.com/sahilganjave05)
