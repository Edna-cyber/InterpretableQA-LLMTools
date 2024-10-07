tools_gpt = [
    {
        "type": "function",
        "function": {
            "name": "Calculator",
            "description": "Conduct an arithmetic operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_query": {
                        "type": "string",
                        "description": "An arithmetic operation containing only numbers and operators, e.g. 2*3.",
                    }
                },
                "required": ["input_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DBLoader",
            "description": "Load a database specified by the DBName and a subset. Normally, we only use DBLoader when the question requires data from a specific structured database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_db": {
                        "type": "string",
                        "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset) and neurips (a papers dataset).",
                    },
                    "duration": {
                        "type": "string",
                        "description": "The subset of the database is specified by a string that evaluates to a list. You must load all the data you need at once. When target_db is hupd, specify the years, e.g. [2012,2013,2015]. The default value is list(range(2004,2019)). When target_db is neurips, specify the rows, e.g. list(range(2000)). The default value is list(range(3585)).",
                    }
                },
                "required": ["target_db", "duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TFIDFMatcher",
            "description": "Find the most relevant document for a given query or to identify pairs of documents that are most relevant to each other",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query or a reference document used for comparison."
                    },
                    "document": {
                        "type": "string",
                        "description": "A document to be compared against the query."
                    }
                },
                "required": ["query", "document"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PandasInterpreter",
            "description": "Interpret Pandas code written in Python and return a dictionary containing the values of variables defined within that code. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use DBLoader before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pandas_code": {
                        "type": "string",
                        "description": "Pandas code written in Python that involves operations on a DataFrame df",
                    }
                },
                "required": ["pandas_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": "Interpret Python code and return a dictionary containing the values of variables defined within that code. Normally, we only use PythonInterpreter when the question requires complex computations. We do not use this tool for tasks that can be performed with Pandas on dataframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "Python code",
                    }
                },
                "required": ["python_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Forecaster",
            "description": "Run a specified forecast model on the previous data to predict the next forecast_len data points",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model_name can be linear_regression or ARIMA",
                    },
                    "previous_data": {
                        "type": "string",
                        "description": "A list of past data points used to train the forecast model",
                    },
                    "forecast_len": {
                        "type": "integer",
                        "description": "The number of data points to be predicted by the forecast model",
                    } 
                },
                "required": ["model_name", "previous_data", "forecast_len"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TextualClassifier",
            "description": "Run a specified binary classifier model on the given textual predictor section to predict the target. Normally, we use the TextualClassifier module for classification tasks that work with textual data as its input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "The database used for the prediction task. The only choices are hupd and neurips."
                    },
                    "model_name": {
                        "type": "string",
                        "description": "The model_name can be logistic_regression, bert-base-uncased, or cnn.",
                    },
                    "section": {
                        "type": "string",
                        "description": "The predictor variable of the classifier model, which is a column that consists of natural language.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The specific instance of text used as input for the predictor variable in the model.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target variable of the classifier model.",
                    },
                    "one_v_all": {
                        "type": "string",
                        "description": "The positive class label for a one-vs-all classification task.",
                    }
                },
                "required": ["database", "model_name", "section", "text", "target", "one_v_all"], 
            },
        },
    }, 
    {
        "type": "function",
        "function": {
            "name": "LLMInferencer",
            "description": "Utilize the current LLM to generate a solution only when the answer cannot be determined from other tools or involves complex reasoning; do not use it for simple type conversions of variable values for the Finish tool. This tool is specifically for addressing uncertainties when these variable values are not explicitly defined or involve random guessing.",
            "parameters": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Terminate the task and return the final answer. Ensure that all variable values are derived DIRECTLY from the output of the previous tool call. If any values require estimation or involve random guessing, leverage the LLMInferencer to accurately determine those values. You MUST USE finish as the final module for solving each question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_values": {
                        "type": "string",
                        "description": "A string that evaluates to a dictionary of variables and their corresponding values, which is exactly the same as the output from the previous tool call. If it leads to an error response, the variable_values must be modified to match the required answer_type, and to match one of the available choices when choices is provided."
                    },
                    "answer_variable": {
                        "type": "string",
                        "description": "A key among the variable_values dictionary that corresponds to the variable which best addresses the question.",
                    },
                    "answer_type": {
                        "type": "string",
                        "description": "A string specifying the required type for the final answer. The only choices are list, float, integer, and string."
                    },
                    "choices": {
                        "type": "string",
                        "description": "A string that evaluates to a list of possible values from which the final answer must be selected."
                    },
                },
                "required": ["variable_values", "answer_variable", "answer_type"], 
            },
        },
    }
]



tools_gemini = [
  {
    "function_declarations": [
      {
        "name": "Calculator",
        "description": "Conduct an arithmetic operation",
        "parameters": {
          "type": "object",
          "properties": {
            "input_query": {
              "type": "string",
              "description": "An arithmetic operation containing only numbers and operators, e.g. 2*3."
            }
          },
          "required": ["input_query"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "DBLoader",
        "description": "Load a database specified by the DBName and a subset. Normally, we only use DBLoader when the question requires data from a specific structured database.",
        "parameters": {
          "type": "object",
          "properties": {
            "target_db": {
              "type": "string",
              "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset) and neurips (a papers dataset)."
            },
            "duration": {
              "type": "string",
              "description": "The subset of the database is specified by a string that evaluates to a list. You must load all the data you need at once. When target_db is hupd, specify the years, e.g. [2012,2013,2015]. The default value is list(range(2004,2019)). When target_db is neurips, specify the rows, e.g. list(range(2000)). The default value is list(range(3585))."
            }
          },
          "required": ["target_db", "duration"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "TestSampler",
        "description": "Shrink the test set to include only the samples specified by the indices",
        "parameters": {
          "type": "object",
          "properties": {
            "indices": {
              "type": "string",
              "description": "The indices from the original test set to be selected. A string in the format of e.g. ID-1,ID-5,ID-1000"
            },
          },
          "required": ["indices"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "TFIDFMatcher",
        "description": "Find the most relevant document for a given query or to identify pairs of documents that are most relevant to each other",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "A query or a reference document used for comparison"
            },
            "document": {
              "type": "string",
              "description": "A document to be compared against the query"
            }
          },
          "required": ["query", "document"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "PandasInterpreter",
        "description": "Interpret Pandas code written in Python and return a dictionary containing the values of variables defined within that code. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use DBLoader before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
        "parameters": {
          "type": "object",
          "properties": {
            "pandas_code": {
              "type": "string",
              "description": "Pandas code written in Python that involves operations on a DataFrame df"
            }
          },
          "required": ["pandas_code"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "PythonInterpreter",
        "description": "Interpret Python code and return a dictionary containing the values of variables defined within that code. Normally, we only use PythonInterpreter when the question requires complex computations. We do not use this tool for tasks that can be performed with Pandas on dataframes.",
        "parameters": {
          "type": "object",
          "properties": {
            "python_code": {
              "type": "string",
              "description": "Python code"
            }
          },
          "required": ["python_code"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "Forecaster",
        "description": "Run a specified forecast model on the previous data to predict the next forecast_len data points.",
        "parameters": {
          "type": "object",
          "properties": {
            "model_name": {
              "type": "string",
              "description": "The model_name can be linear_regression or ARIMA."
            },
            "previous_data": {
              "type": "string",
              "description": "A list of past data points used to train the forecast model."
            },
            "forecast_len": {
              "type": "integer",
              "description": "The number of data points to be predicted by the forecast model."
            }
          },
          "required": ["model_name", "previous_data", "forecast_len"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "TextualClassifier", 
        "description": "Run a specified binary classifier model on the given textual predictor section to predict the target. Normally, we use the TextualClassifier module for classification tasks that work with textual data as its input.",
        "parameters": {
          "type": "object",
          "properties": {
            "database": {
              "type": "string",
              "description": "The database used for the prediction task. The only choices are hupd and neurips."
            },
            "model_name": {
              "type": "string",
              "description": "The model_name can be logistic_regression, bert-base-uncased, or cnn."
            },
            "section": {
              "type": "string",
              "description": "The predictor variable of the classifier model, which is a column that consists of natural language."
            },
            "text": {
              "type": "string",
              "description": "The specific instance of text used as input for the predictor variable in the model."
            },
            "target": {
              "type": "string",
              "description": "The target variable of the classifier model."
            },
            "one_v_all": {
              "type": "string",
              "description": "The positive class label for a one-vs-all classification task."
            }
          },
          "required": ["database", "model_name", "section", "text", "target", "one_v_all"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "LLMInferencer",
        "description": "Utilize the current LLM to generate a solution only when the answer cannot be determined from other tools or involves complex reasoning; do not use it for simple type conversions of variable values for the Finish tool. This tool is specifically for addressing uncertainties when these variable values are not explicitly defined or involve random guessing."
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "Finish",
        "description": "Terminate the task and return the final answer. Ensure that all variable values are derived DIRECTLY from the output of the previous tool call. If any values require estimation or involve random guessing, leverage the LLMInferencer to accurately determine those values. You MUST USE finish as the final module for solving each question.",
        "parameters": {
          "type": "object",
          "properties": {
            "variable_values": {
              "type": "string",
              "description": "A string that evaluates to a dictionary of variables and their corresponding values, which is exactly the same as the output from the previous tool call. If it leads to an error response, the variable_values must be modified to match the required answer_type, and to match one of the available choices when choices is provided."
            },
            "answer_variable": {
              "type": "string",
              "description": "A key among the variable_values dictionary that corresponds to the variable which best addresses the question."
            },
            "answer_type": {
              "type": "string",
              "description": "A string specifying the required type for the final answer. The only choices are list, float, integer, and string."
            },
            "choices": {
              "type": "string",
              "description": "A string that evaluates to a list of possible values from which the final answer must be selected."
            },
          },
          "required": ["variable_values", "answer_variable", "answer_type"]
        }
      }
    ]
  }
]