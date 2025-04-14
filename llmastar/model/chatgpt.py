import openai
import os
import asyncio

# این کلاس برای ارتباط با API GPT طراحی شده است
class ChatGPT:
    def __init__(self, method, sysprompt, example):
        self.id = 0
        self.chat_history = [{"role": "system", "content": sysprompt}]
        self.cache = {}  # کش برای ذخیره نتایج

        # در صورتی که مثال‌هایی برای ورودی‌ها وجود داشته باشد، آن‌ها را پردازش می‌کنیم
        if example:
            self.prompt = sysprompt + f'\nFollow these examples delimited with “”" as a guide.\n'
            keys = list(example.keys())
            for key in keys:
                input = example[key]
                index = input.find("\n") + 1
                self.prompt += f'“”"\nUser: {input[:index - 1]}\nAssistant: {input[index:]}“”"\n'
                self.chat_history.append({"role": "user", "content": input[:index - 1]})
                self.chat_history.append({"role": "assistant", "content": input[index:]})
        else:
            self.prompt = sysprompt

    def ask(self, prompt, stop=["\n"], max_tokens=100):
        """
        این متد درخواست را از کش چک می‌کند، اگر قبلاً پردازش شده باشد، از کش استفاده می‌کند.
        در غیر این صورت، درخواست را به OpenAI ارسال می‌کند.
        """
        if prompt in self.cache:
            return self.cache[prompt]

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # می‌توان مدل را تغییر داد تا سبک‌تر باشد
            prompt=prompt,
            temperature=0,  # کاهش تصادفی بودن برای سرعت بیشتر
            top_p=0.9,  # محدود کردن فضای جستجو برای سرعت بیشتر
            max_tokens=max_tokens,
            stop=stop
        )
        result = response["choices"][0]["text"]
        self.cache[prompt] = result  # ذخیره نتیجه در کش
        return result

    def chat(self, query, prompt="", stop=["\n"], max_tokens=100):
        """
        این متد برای ارسال درخواست‌های چت به OpenAI است.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": prompt}
            ],
            temperature=0,  # کاهش تصادفی بودن برای سرعت بیشتر
            max_tokens=max_tokens,
            stop=stop
        )
        return response["choices"][0]["message"]["content"]

    async def ask_async(self, prompt, stop=["\n"], max_tokens=100):
        """
        این متد غیرهمزمان برای ارسال درخواست به OpenAI است.
        """
        response = await openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,  # کاهش تصادفی بودن برای سرعت بیشتر
            top_p=0.9,
            max_tokens=max_tokens,
            stop=stop
        )
        return response["choices"][0]["text"]

    async def batch_ask(self, prompts):
        """
        ارسال چندین درخواست به صورت غیرهمزمان
        """
        tasks = [self.ask_async(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def chat_with_image(self, chat_history, stop=["\n"], max_tokens=100):
        """
        این متد برای ارسال درخواست‌های چت همراه با تصویر به OpenAI است.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            temperature=0,
            max_tokens=max_tokens,
            stop=stop
        )
        return response["choices"][0]["message"]["content"]

