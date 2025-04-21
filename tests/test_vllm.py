from coreai.infer.qwenvl_vllm import QwenVLvLLMInfer
import asyncio

async def main():
    model_p = 'checkpoints/Qwen2.5-VL-3B-Instruct'
    model = QwenVLvLLMInfer(model_path=model_p)
    
    a = await model.generate_single_turn(
        '<HIS><image><image><image><image></HIS><OBS><image></OBS><NAV>go to sofa', [
        'data/images/000000/000.png',
        'data/images/000000/001.png',
        'data/images/000000/002.png',
        'data/images/000000/003.png',
        'data/images/000000/004.png'],
        system_msg='you are a nagivation robot',
        verbose=True
    )
    
    print(a)

if __name__ == '__main__':
    asyncio.run(main())
    