# 예제1
# https://pytorch.org/get-started/locally/
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# conda가 잘 돌아가는지 테스트하기 위한 예제
# import torch
# x = torch.rand(5, 3) 
# print(x)

# print(torch.cuda.is_available())

# 예제 2 vision pipeline
# https://huggingface.co/nateraw/vit-base-cats-vs-dogs
# from transformers import pipeline

# vision_classifier = pipeline(model="runwayml/stable-diffusion-v1-5")
# # vision_classifier = pipeline(model="nateraw/vit-base-cats-vs-dogs", device="cpu")
# # preds = vision_classifier(
# #     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# # )
# # preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
# preds = vision_classifier(
#     images="./cat1.jpg"
# )
# preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
# print(preds)

# 예제 3
# https://huggingface.co/docs/diffusers/main/en/installation 대로 설치
# https://huggingface.co/runwayml/stable-diffusion-v1-5
# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on earth"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse_earh.png")

# 예제 4
# https://huggingface.co/damo-vilab/text-to-video-ms-1.7b
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is underwater"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)
print(video_path)

# pytorch를 main으로 함
# tip : git bash로 리눅스 명령어 입력 가능함
# 가상환경 만들것
    # conda create -n pro1 python=3.10                                                          // 콘다 설치
    # conda activate pro1                                                                       // 콘다 활성화
    # conda env list                                                                            // 환경설정 확인
    # pytorch install                                                                           // https://pytorch.org/get-started/locally/
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia       // pytorch 설치
        # y / n 에서 y 입력
    # main.py 실행 테스트, gpu 출력 여부까지 확인해야함, import 부분이 vs code에서는 코드에디터와 cli가 별개라서 인식 문제가 있음
    # 탭 하단 3.10.12(base 부분 확인, conda로 갱신해야됨)
    # cli에서 True 결과값 확인하면 됨
    # hugging face -> pip install transformers
    # 딥러닝을 몰라도 task와 문서를 읽을수만 있으면 가능함
    # 많은 task들을 시도해보는 것이 중요
# pipeline에는 task와 model이 들어감, pipeline은 문서를 본 누구나 사용가능
# 학습 -> 추론 -> 포스트 프로세싱 이 모든 과정을 파이프라인에서는 처리함
# 모든 task가 파이프라인이 존재하는 것은 아님
# 난이도는 파이프라인이 낮음, task, model은 반드시 알고 있어야함
# 파이프라인이 구현되지 않은 task만 AutoModel 적용!