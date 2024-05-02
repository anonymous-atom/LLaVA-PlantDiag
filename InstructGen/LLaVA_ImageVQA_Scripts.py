#!/usr/bin/env python
# coding: utf-8




pip install --upgrade openai


# In[8]:


get_ipython().system("export OPENAI_API_KEY='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'")


# In[ ]:


get_ipython().system('unzip /dataset.zip')


# In[10]:


prompt = """You are an AI assistant specialized in
plant disease topics.
You are provided with a string (Plant Name and Leaf Disease associated with image of the leaf) of a figure image from a plant leaf disease
dataset. Unfortunately, you don’t have access to the actual image.
Below are requirements for generating the questions and answers in the conversation:
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or
names, as these may reveal the conversation is based on the text information, rather than
the image itself. Focus on the visual aspects of the image that can be inferred without
the text information.
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead,
refer to the information as being "in the image."
- Ensure that questions are diverse and cover a range of visual aspects of the image.
- The conversation should include at least 2-3 turns of questions and answers about the
visual aspects of the image.
- Answer responsibly, avoiding overconfidence, and do not provide medical advice or
diagnostic information. Encourage the user to consult a healthcare professional for
advice.
Produce Output in JSON Format"""


# In[4]:


dataset_directory = "/plantvillage_dataset/color"


# In[14]:


from openai import OpenAI
client = OpenAI(api_key = "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")

completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": prompt},
    {"role": "user", "content": "Apple_AppleScab"}
  ],
  # response_format= { "type":"json_object" }
)

print(completion)


# In[15]:


print(completion.choices[0].message.content)


# In[5]:


caption_extend_prompt = """You are an AI assistant specialized in plant disease topics.
You are provided with a string (Plant Name and Leaf Disease associated with image of the leaf) of a figure image from a plant leaf disease
dataset. Unfortunately, you don’t have access to the actual image. You have to generate 30 to 50 words caption using that string.
Below are the important requirements you must consider:
1. Don't make the caption sounds like as if you don't have access to image.
2. Write the caption so it sounds like you are looking at the image.
3. Make sure the caption doesn't sounds uncertain.
4. Generate confident captions."""


# In[6]:


from openai import OpenAI
client = OpenAI(api_key = "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")

def extend_caption(name):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": caption_extend_prompt },
        {"role": "user", "content": name}
      ],
      # response_format= { "type":"json_object" }
    )

    return completion.choices[0].message.content

# extend_caption()


# In[23]:


import os
dataset_directory = "/content/content/plantvillage_dataset/color"

def check_out(directory=dataset_directory, max_images_per_folder=1):
    folder_data = []  # List to store dictionaries for each folder

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            folder_dict = {}  # Dictionary for the current folder
            uploaded_count = 0

            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.JPG'):  # Adjust the file extension as needed
                    if uploaded_count >= max_images_per_folder:
                        break  # Stop uploading once the limit is reached

                    image_path = os.path.join(folder_path, filename)
                    print(filename)
                    uploaded_count += 1

            folder_data.append({folder_name: folder_dict})

    return []

check_out()


# In[9]:


import os

def return_x(img_name):
    x = {}
    base_name, extension = os.path.splitext(os.path.basename(img_name))
    x['org_name'] = img_name
    x['pair_id'] = base_name
    x['end'] = extension # Exclude the leading dot from the extension
    return x


# In[26]:





# In[10]:


import random

data = []

def format_conv(x, assistant_msg):

  image_description_actions = [
      "Narrate the contents of the image with precision",
      "Illustrate the image through a descriptive explanation",
      "Share a comprehensive rundown of the presented image",
      "Present a compact description of the photo’s key features",
      "Give an elaborate explanation of the image you see",
      "Relay a brief, clear account of the picture shown",
      "Examine the image closely and share its details",
      "Describe the image concisely",
      "Clarify the contents of the displayed image with great detail",
      "Analyze the image in a comprehensive and detailed manner",
      "Write an exhaustive depiction of the given image",
      "Walk through the important details of the image",
      "Describe the following image in detail",
      "Summarize the visual content of the image",
      "Share a concise interpretation of the image provided",
      "Provide a brief description of the given image",
      "Create a compact narrative representing the image presented",
      "Explain the various aspects of the image before you",
      "Portray the image with a rich, descriptive narrative",
      "Provide a detailed description of the given image",
      "Characterize the image using a well-detailed description",
      "Give a short and clear explanation of the subsequent image",
      "Offer a succinct explanation of the picture presented",
      "Render a clear and concise summary of the photo",
      "Break down the elements of the image in a detailed manner",
      "Write a terse but informative summary of the picture",
      "Offer a thorough analysis of the image"
  ]

  user_msg = random.choice(image_description_actions)

  conversations = []
  conversations.extend([
    {
      'from': 'human',
      'value': user_msg,
    },
    {
      'from': 'gpt',
      'value': assistant_msg,
    }
  ])

  return {
    'id': x['pair_id'],
    'image': x['org_name'],
    'conversations': conversations,
    }


# In[11]:


import os
dataset_directory = "/content/content/plantvillage_dataset/color"

data = []
def check_out(directory=dataset_directory, max_images_per_folder=1):

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            folder_dict = {}  # Dictionary for the current folder
            uploaded_count = 0

            for filename in os.listdir(folder_path):

                if filename.endswith('.jpg') or filename.endswith('.JPG'):  # Adjust the file extension as needed
                    if uploaded_count >= max_images_per_folder:
                        break  # Stop uploading once the limit is reached

                    image_path = os.path.join(folder_path, filename)
                    uploaded_count += 1
                    x = return_x(filename)
                    assistant_msg = "test" # extend_caption(folder_name) #Make request to GPT to generate caption
                    data.append(format_conv(x, assistant_msg))

    return data

data = check_out(dataset_directory, 200)


# In[12]:


len(data)


# In[13]:


data


# In[31]:


import os
import asyncio
import aiohttp
from openai import AsyncOpenAI

dataset_directory = "/content/content/plantvillage_dataset/color"

client = AsyncOpenAI(api_key="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")  # Replace "My API Key" with your actual OpenAI API key

async def extend_caption_async(name):
    async with aiohttp.ClientSession() as session:
        async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": caption_extend_prompt},
                        {"role": "user", "content": name},
                    ],
                },
                headers={"Authorization": f"Bearer {client.api_key}"},
        ) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]

async def process_folder(folder_path, max_images_per_folder):
    folder_dict = {}  # Dictionary for the current folder
    uploaded_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.JPG'):  # Adjust the file extension as needed
            if uploaded_count >= max_images_per_folder:
                break  # Stop uploading once the limit is reached

            image_path = os.path.join(folder_path, filename)
            uploaded_count += 1
            x = return_x(filename)
            assistant_msg = await extend_caption_async(folder_path)  # Make request to GPT to generate caption
            data.append(format_conv(x, assistant_msg))

async def check_out(directory=dataset_directory, max_images_per_folder=1):
    tasks = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            tasks.append(process_folder(folder_path, max_images_per_folder))

    await asyncio.gather(*tasks)

data = []
await check_out(dataset_directory, 200)


# In[37]:


len(data)


# In[38]:


data


# In[34]:


import json

# Convert data to JSON
json_data = json.dumps(data, indent=2)

# Save JSON to a file
with open('output.json', 'w') as json_file:
    json_file.write(json_data)


# In[ ]:




