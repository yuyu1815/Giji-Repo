import toml
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


class AI_Chat():
    def __init__(self, model_name:str=None,max_tokens:int=8000):
        #マックストークン8000に仮設定
        chat_config = toml.load(open("config.toml", encoding="utf-8"))
        if not (model_name or chat_config["MODEL_NAME"]):
            print ("モデルの名前が設定されていません")
            exit(1)
        if not chat_config["OPEN_AI_API_KEY"]:
            print("OPENAI_API_KEY が設定されていません")
            exit(1)
        self.max_tokens = min(max_tokens, 8000)
        self.model_name = model_name or chat_config["MODEL_NAME"]
        if "gemini" in self.model_name:
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=chat_config["GEMINI_AI_API_KEY"],
                                              temperature=chat_config["TEMPERATURE"], max_tokens=self.max_tokens)
        elif "claude" in self.model_name:
            self.llm = ChatAnthropic(model=self.model_name, api_key=chat_config["CLAUD_AI_API_KEY"],
                                     temperature=chat_config["TEMPERATURE"], max_tokens=self.max_tokens)
        elif "GPT" in self.model_name:
            self.llm = OpenAI(model=self.model_name, openai_api_key=chat_config["OPEN_AI_API_KEY"],
                                        temperature=chat_config["TEMPERATURE"],max_tokens=self.max_tokens)
        else:
            print(f"指定されたモデル名 '{model_name}' 未対応です")
            exit(1)
    def chat(self,msg:str)->str:
        """
        :param msg: 本文
        :return:
        """
        return self.llm.invoke(msg)

    def prompt_Chat(self,prompt_msg:str,text:str)->str:
        """
        :param prompt_msg: テンプレで聞くこと
        :param text: 本文
        :return:
        """
        template = """{prompt_msg} {text}"""
        prompt = PromptTemplate(template=template, input_variables=["prompt_msg", "text"])
        chain = prompt | self.llm
        return chain.invoke({"prompt_msg": prompt_msg, "text": text}).content
