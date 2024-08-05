'''
input: formula strings
output: the answer of the mathematical formula
'''
import sympy as sp

def calculator(input_query):
    try:
        expr = sp.sympify(input_query)
        return {"calculator result": expr.evalf()}
    except TypeError as e:
        return "Error: input_query must contain only numbers and operators."
    except Exception as e:
        return "Error: "+str(e)

if __name__ == "__main__":
    input_query = '3 + 5 * (8 - 9) * 4 - 6 / 20'
    input_query = 'len(df_2004) / len(df_2005)'
    print(calculator(input_query))