from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage,ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gwen3:1.7b"

#------ Tools (Langchain @tool decorator) ------#

@tool
def get_product_price(product:str)->float:
    """Look up the price of a product in the catalog."""
    print(f" >> Execuitng get_product_price(product={product})")
    prices = {
        "laptop": 999.99,
        "headphones": 199.99,
        "keyboard": 49.99,}
    return prices.get(product, 0)

@tool
def apply_discount(price:float,discount_tier:str)->float:
    """"Apply a discount tier to a price and return the final price.
        Available tiers:bronze, silver, gold"""
    print(f" >> Executing apply_discount(price={price},discount_tier={discount_tier})")
    discount_percentages = {
        "bronze":5,"silver":12,"gold":23
        }
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

#------ Agent Loop ------#
@traceable(name="langchain-agent-loop")
def run_agent(question:str):
    pass

if __name__ == "__main__":
    print("Hello LangChain Agents (.bind_tools)")
    print()
    result= run_agent("What is the price of a laptop after applying a gold discount?")