import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import vertexai.preview.generative_models as generative_models
import re
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason, Tool
from utils.config import cf

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]


def get_google_gemini_generate_answer(question=''):
    try:
        if not question:
            return

        vertexai.init(project=cf.get("PROJECT_ID"), location="us-central1")
        model = GenerativeModel(
            "gemini-1.5-flash-001",
        )
        responses = model.generate_content(
            [question],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        result = ""
        for response in responses:
            if response:
                result += response.text

        print('result', result)

        return re.sub("\s\s+", " ", result.strip())
    except Exception as e:
        # logger.info(
        #     "failed to get google_gemini_generate_answer, question=%s err=%s", question, e)
        return ""


def google_gemini_generate_answer_with_grounding(question=''):
    result = ""
    try:
        if not question:
            return
        vertexai.init(project="anfinx-prod", location="us-central1")
        tools = [
            Tool.from_google_search_retrieval(
                google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval()
            ),
        ]
        model = GenerativeModel(
            "gemini-1.5-pro-001",
            tools=tools,
        )
        responses = model.generate_content(
            [question],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        for response in responses:
            if response.candidates[0].finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED:
                continue
            result += response.text
        return re.sub("\s\s+", " ", result.strip())
    except Exception as e:
        # print(f'failed to get google_gemini_generate_answer_with_grounding')
        return re.sub("\s\s+", " ", result.strip())
