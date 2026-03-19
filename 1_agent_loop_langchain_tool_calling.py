from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage,ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

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
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name:t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
   
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("="*50)
    messages = [
        SystemMessage(
        content = (
            "You are a helpful shpopping assistant."
            "You have access to the product catalog tool and a discount tool.\n\n"
            "STRICT RULES - you must these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number.\n"
            "3. NEVER calculate the discount yourself using math- ALWAYS use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.\n\n"
        )),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = llm_with_tools.invoke(messages)

        total_calls = ai_message.tool_calls

        if not total_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        total_call = total_calls[0]
        tool_name = total_call.get("name")
        tool_args = total_call.get("args",{})
        total_call_id = total_call.get("id")

        print(f"[Tool Selected] {tool_name} with args:  {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation} ")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation),tool_call_id=total_call_id))
    print("Max iterations reached without a final answer.")
    return None




if __name__ == "__main__":
    print("Hello LangChain Agents (.bind_tools)")
    print()
    result= run_agent("What is the price of a laptop after applying a gold discount?")