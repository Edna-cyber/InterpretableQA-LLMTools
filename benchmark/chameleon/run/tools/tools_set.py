tools_gpt = [
    {
        "type": "function",
        "function": {
            "name": "Calculate",
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
            "name": "LoadDB",
            "description": "Load a database specified by the DBName, train and test subsets, and a column to be predicted. Normally, we only use LoadDB when the question requires data from a specific structured database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_db": {
                        "type": "string",
                        "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset) and neurips (a papers dataset).",
                    },
                    "train_duration": {
                        "type": "string",
                        "description": "The training subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2004-2006. When target_db is neurips, specify the range of rows in the format 0-endRow, e.g. 0-2000. When the task does not involve prediction and the target_db is neurips, use the default range 0-3585.",
                    },
                    "test_duration": {
                        "type": "string",
                        "description": "The testing subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2016-2018. When target_db is neurips, specify the range of rows in the format startRow-3585, e.g. 2001-3585, where startRow must be one more than the endRow of train_duration. When the task does not involve prediction, set this value to None.",
                    },
                    "outcome_col": {
                        "type": "string",
                        "description": "The column to predict if the task involves making a prediction. If no prediction is required, set this value to None.",
                    }
                },
                "required": ["target_db", "train_duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TFIDF",
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
            "description": "Interpret Pandas code written in Python and return a dictionary containing the values of variables defined within that code. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use LoadDB before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
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
            "description": "Run a specified classifier model on the given textual predictorSection to predict the target. Normally, we use the TextualClassifier module for classification tasks that work with textual data as its input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "The name of the database that this classification task is conducted on. The only choices are hupd and neurips."
                    },
                    "model_name": {
                        "type": "string",
                        "description": "The model_name can be logistic_regression, distilbert-base-uncased, cnn, or naive_bayes.",
                    },
                    "section": {
                        "type": "string",
                        "description": "The predictor variable of the classifier model, which is a column that consists of natural language requiring tokenization.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target variable of the classifier model.",
                    },
                    "one_v_all": {
                        "type": "string",
                        "description": "The class label for a one-vs-all classification task. When it's set to default value None, the model will predict all possible classes.",
                    }
                },
                "required": ["database", "model_name", "section", "target"], 
            },
        },
    }, 
    {
        "type": "function",
        "function": {
            "name": "LLMInterpreter",
            "description": "Use the current LLM to generate an answer.",
            "parameters": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Terminate the task and return the final answer. You must use Finish as the final module for solving each question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_values": {
                        "type": "string",
                        "description": "A string that evaluates to a dictionary of variables and their corresponding values.",
                    },
                    "answer_variable": {
                        "type": "string",
                        "description": "A key among the variable_values dictionary that corresponds to the variable which best addresses the question.",
                    },
                    "answer_type": {
                        "type": "string",
                        "description": "A string specifying the required type for the final answer. The only choices are list, float, integer, and string."
                    }
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
        "name": "Calculate",
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
        "name": "LoadDB",
        "description": "Load a database specified by the DBName, train and test subsets, and a column to be predicted. Normally, we only use LoadDB when the question requires data from a specific structured database.",
        "parameters": {
          "type": "object",
          "properties": {
            "target_db": {
              "type": "string",
              "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset) and neurips (a papers dataset)."
            },
            "train_duration": {
              "type": "string",
              "description": "The training subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2004-2006. When target_db is neurips, specify the range of rows in the format 0-endRow, e.g. 0-2000. When the task does not involve prediction and the target_db is neurips, use the default range 0-3585."
            },
            "test_duration": {
              "type": "string",
              "description": "The testing subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2016-2018. When target_db is neurips, specify the range of rows in the format startRow-3585, e.g. 2001-3585, where startRow must be one more than the endRow of train_duration. When the task does not involve prediction, set this value to None."
            },
            "outcome_col": {
              "type": "string",
              "description": "The column to predict if the task involves making a prediction. If no prediction is required, set this value to None."
            }
          },
          "required": ["target_db", "train_duration"]
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
        "name": "TFIDF",
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
        "description": "Interpret Pandas code written in Python and return a dictionary containing the values of variables defined within that code. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use LoadDB before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
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
        "description": "Run a specified classifier model on the given textual predictor section to predict the target. Normally, we use the TextualClassifier module for classification tasks that work with textual data as its input.",
        "parameters": {
          "type": "object",
          "properties": {
            "model_name": {
              "type": "string",
              "description": "The model_name can be logistic_regression, distilbert-base-uncased, cnn, or naive_bayes."
            },
            "section": {
              "type": "string",
              "description": "The predictor variable of the classifier model, which is a column that consists of natural language requiring tokenization."
            },
            "target": {
              "type": "string",
              "description": "The target variable of the classifier model."
            },
            "one_v_all": {
              "type": "string",
              "description": "The class label for a one-vs-all classification task. When it's set to default value None, the model will predict all possible classes."
            }
          },
          "required": ["model_name", "section", "target"]
        }
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "LLMInterpreter",
        "description": "Use the current LLM to generate an answer."
      }
    ]
  },
  {
    "function_declarations": [
      {
        "name": "Finish",
        "description": "Terminate the task and return the final answer. You must use Finish as the final module for solving each question.",
        "parameters": {
          "type": "object",
          "properties": {
            "variable_values": {
              "type": "string",
              "description": "A string that evaluates to a dictionary of variables and their corresponding values."
            },
            "answer_variable": {
              "type": "string",
              "description": "A key among the variable_values dictionary that corresponds to the variable which best addresses the question."
            },
            "answer_type": {
              "type": "string",
              "description": "A string specifying the required type for the final answer. The only choices are list, float, integer, and string."
            }
          },
          "required": ["variable_values", "answer_variable", "answer_type"]
        }
      }
    ]
  }
]