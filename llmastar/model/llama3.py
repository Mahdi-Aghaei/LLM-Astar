import transformers
import torch

class Llama3:
    def __init__(self):
        # استفاده از مدل و pipeline
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # استفاده از float16 برای بهبود سرعت
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},  # از float16 استفاده می‌کنیم که معمولاً سریعتر است
            device=0,  # اگر از یک GPU خاص استفاده می‌کنید، GPU را انتخاب کنید
            batch_size=4  # تعداد نمونه‌های همزمان
        )
        
        # تعریف ترمیناتورها
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # این باید برای اتمام توکن استفاده شود
        ]
    
    def ask(self, prompt):
        # استفاده از batching و تنظیمات بهینه برای زمان پردازش سریع‌تر
        outputs = self.pipeline(
            prompt,
            max_new_tokens=1000,  # تعداد توکن‌های تولید شده را کاهش دهید تا سرعت بهبود یابد
            eos_token_id=self.terminators,
            do_sample=False,  # از sample کردن استفاده نمی‌کنیم تا سرعت بالا برود
            temperature=0,  # تنظیم temperature به صفر تا تنوع کمتری داشته باشیم
            top_p=1,  # محدودیت توزیع احتمال بیشتر
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][len(prompt):]

    # اگر بخواهید پردازش همزمان را پیاده‌سازی کنید
    def ask_batch(self, prompts):
        """
        پردازش همزمان چندین درخواست برای سرعت بیشتر
        """
        outputs = self.pipeline(
            prompts,
            max_new_tokens=1000,
            eos_token_id=self.terminators,
            do_sample=False,
            temperature=0,
            top_p=1,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return [output["generated_text"][len(prompt):] for prompt, output in zip(prompts, outputs)]

    def optimize_model(self):
        """
        جیت‌کامپایل کردن مدل با استفاده از torch.jit برای بهبود سرعت.
        """
        self.pipeline.model = torch.jit.script(self.pipeline.model)  # استفاده از JIT برای بهینه‌سازی
