from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#------ Tools (Langchain @tool decorator) ------#

@traceable(type="tool", name="get_product_price")
def get_product_price(product:str)->float:
    """Look up the price of a product in the catalog."""
    print(f" >> Execuitng get_product_price(product={product})")
    prices = {
        "laptop": 999.99,
        "headphones": 199.99,
        "keyboard": 49.99,}
    return prices.get(product, 0)

@traceable(type="tool", name="apply_discount")
def apply_discount(price:float,discount_tier:str)->float:
    """"Apply a discount tier to a price and return the final price.
        Available tiers:bronze, silver, gold"""
    print(f" >> Executing apply_discount(price={price},discount_tier={discount_tier})")
    discount_percentages = {
        "bronze":5,"silver":12,"gold":23
        }
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

tools_for_llm = [
    {
        "type":"function",
        "function":{
            "name":"get_product_price",
            "description":"Look up the price of a product in the catalog.",
            "parameters":{
                "type":"object",
                "properties":{
                    "product":{
                        "type":"string",
                        "description":"The name of the product to look up the price for."
                    },
                },
                "required":["product"]
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"apply_discount",
            "description":"Apply a discount tier to a price and return the final price. Available tiers:bronze, silver, gold",
            "parameters":{
                "type":"object",
                "properties":{
                    "price":{
                        "type":"number",
                        "description":"The price to apply the discount to."
                    },
                    "discount_tier":{
                        "type":"string",
                        "description":"The discount tier to apply. One of bronze, silver, or gold."
                    },
                },
                "required":["price","discount_tier"]
            },
        },
    }
]

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL,tools=tools_for_llm,messages=messages)

#------ Agent Loop ------#
@traceable(name="ollama-agent-loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }

    print(f"Question: {question}")
    print("="*50)
    messages = [
        {"role" :"system",
         "content" : (
            "You are a helpful shpopping assistant."
            "You have access to the product catalog tool and a discount tool.\n\n"
            "STRICT RULES - you must these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number.\n"
            "3. NEVER calculate the discount yourself using math- ALWAYS use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.\n\n"
        )},
        {"role":"user","content":question},
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        

        total_calls = ai_message.tool_calls

        if not total_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        total_call = total_calls[0]
        tool_name = total_call.function.name
        tool_args = total_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args:  {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use(**tool_args)
        print(f"[Tool Result] {observation} ")

        messages.append(ai_message)
        messages.append(
            {
                "role":"tool",
                "content":str(observation)
            }
        )
    print("Max iterations reached without a final answer.")
    return None




if __name__ == "__main__":
    print("Hello LangChain Agents (.bind_tools)")
    print()
    result= run_agent("What is the price of a laptop after applying a gold discount?")