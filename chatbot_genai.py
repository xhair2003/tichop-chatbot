from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("API key for GEMINI is not set. Please check your .env file.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load product data
file_path = 'D:\\Code_Space\\CHATBOT\\chatbott\\Sales_Product_Details.csv'

# Load CSV và loại bỏ cột trùng lặp
try:
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.duplicated()]  # Loại bỏ các cột trùng
except FileNotFoundError:
    raise ValueError(f"CSV file not found at {file_path}")
except Exception as e:
    raise ValueError(f"Error loading CSV file: {str(e)}")

# Kiểm tra xem các cột bắt buộc có tồn tại không
required_columns = ['Product_Description', 'Product_Category', 'Product_Line', 'Raw_Material', 'Unit_Price', 'Seasonal_Trend']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing from the CSV file: {missing_columns}")

# Chuyển đổi dữ liệu thành dictionary
product_data = df[required_columns].drop_duplicates().set_index('Product_Description').T.to_dict()

def search_product(keyword: str) -> str:
    """Search for products matching the keyword."""
    matching_products = []
    keyword_lower = keyword.lower()
    for product_name, product in product_data.items():
        if (keyword_lower in product_name.lower() or
            keyword_lower in product['Product_Category'].lower() or
            keyword_lower in product['Product_Line'].lower() or
            keyword_lower in product['Raw_Material'].lower() or
            keyword_lower in str(product['Unit_Price']).lower() or
            keyword_lower in product['Seasonal_Trend'].lower()):
            description = (f"{product_name} - {product['Product_Category']} "
                           f"({product['Product_Line']}, Material: {product['Raw_Material']}, "
                           f"Price: {round(product['Unit_Price'], 2)}, Trend: {product['Seasonal_Trend']})")
            matching_products.append(description)
    if matching_products:
        return "\n".join(matching_products)
    else:
        return "No matching products found."

@app.route("/chat", methods=["POST"])
def chat_with_user():
    """Generate chatbot response based on user input."""
    try:
        # Lấy input từ yêu cầu JSON
        user_input = request.json.get("user_input", "")

        # Search for product in the data
        product_response = search_product(user_input)
        
        # Kiểm tra nếu có sản phẩm phù hợp
        if product_response != "No matching products found.":
            # Tạo câu trả lời dựa trên sản phẩm
            response_text = f"We found some matching products for your request:\n\n{product_response}\n\nWould you like more details about any of these products?"
        else:
            # Nếu không có sản phẩm, sử dụng Generative AI
            system_instruction = """
            You are Min, a friendly and knowledgeable assistant specializing in women's clothing. Your role is to help customers find the perfect items, process their orders, and offer valuable fashion advice. You have access to detailed product inventory data, stored in a CSV file, which includes information such as product name, category, product line, material, price, available sizes, colors, and seasonal trends.

            Store Information:
            Operating Hours: Monday to Friday: 9 AM - 7 PM; Saturday to Sunday: 10 AM - 6 PM. (You have access to real-time Việt Nam (GMT+7))
            Your priority is to provide accurate, helpful, and engaging responses.
            Your Goals:
            Assist Customers:

            Help customers browse available products.
            Provide detailed information about pricing, materials, styles, sizes, colors, and seasonal suitability.
            Suggest items based on customer preferences, needs, and current trends.
            Retrieve Data Accurately:

            Use the product inventory CSV file to ensure your responses are based on real data.
            Confirm product availability, specifications, and seasonal trends to align with customer requests.
            Make Fashion Recommendations:

            Suggest outfit combinations or matching accessories based on product details.
            Provide style tips and advice based on the current season or special events.
            Ensure Customer Satisfaction:

            Double-check all order details with the customer before finalizing.
            Respond naturally and warmly to greetings or general queries.
            If a customer’s request is unclear, kindly ask for more details to ensure accurate assistance.
            Example Scenarios:
            When a customer asks about a specific product:
            "The casual shirt you’re looking for is part of our Womenswear category, made from 100% Cotton, and is priced at $32.27. It’s perfect for Winter and comes in navy blue and white. Would you like me to check if your size is available?"

            When a customer wants outfit suggestions:
            "For a summer outing, you might love our lightweight sundresses in floral prints. They pair beautifully with our wide-brim hats and casual sandals. Shall I suggest a color or size for you?"

            When a customer inquires about trends:
            "This season, pastel tones and airy fabrics are very popular. Our collection includes soft linen blouses and flowing skirts that are ideal for the warmer months. Would you like to explore these options?"
            """

            # Generative AI configuration
            generation_config = {
                "temperature": 2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            # Create a chat session
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
                system_instruction=system_instruction,
            )
            chat_session = model.start_chat(history=[{"role": "user", "parts": [user_input]}])

            # Generate response
            ai_response = chat_session.send_message(user_input)
            response_text = ai_response.text

        # Trả về phản hồi
        return jsonify({"response": response_text, "product_info": product_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
