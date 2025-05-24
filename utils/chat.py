import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import vertexai.preview.generative_models as generative_models
import re
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason, Tool
from utils.config import cf
from google import genai
from google.genai import types
import base64


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
    # try:
        if not question:
            return
        print('cf', cf)

        vertexai.init(project=cf.get("PROJECT_ID"), location="global")
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
    # except Exception as e:
        # print(
        #     "failed to get google_gemini_generate_answer, question=%s err=%s", question, e)
        # return ""


def get_google_gemini_generate_answer_v2(question=''):
    client = genai.Client(
        vertexai=True,
        project="intelligent-tutoring-system",
        location="global",
    )

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=question)
        ]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
    )

    result = ""
    for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        result += chunk.text
    return re.sub("\s\s+", " ", result.strip())


def google_gemini_generate_answer_with_grounding(question=''):
    result = ""
    try:
        if not question:
            return
        vertexai.init(project="intelligent-tutoring-system", location="us-central1")
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


def get_response_from_table_gpt(text):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "tablegpt/TableGPT2-7B"

    # Load the model (with device_map="auto" and offloading enabled)
    table_gpt_model = AutoModelForCausalLM.from_pretrained(
        model_name,  # Replace with your model name
        torch_dtype="auto",  # Auto-detect precision
        device_map="auto",  # Automatically assign devices (GPU/CPU)
    )

    # Specify the offload directory (the folder where the model weights will be saved)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_inputs = tokenizer([text], return_tensors="pt").to(table_gpt_model.device)
    generated_ids = table_gpt_model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for output_ids, input_ids in zip(model_inputs["input_ids"], generated_ids)]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)[0]
    return response
