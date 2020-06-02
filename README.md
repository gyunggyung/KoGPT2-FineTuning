# KoGPT2-FineTuning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qfL-IUp4k0uzkr_6SIaAmS_PA_Luvt1t)
[![license Apache-2.0](https://img.shields.io/badge/license-Apache2.0-red.svg?style=flat)](https://github.com/gyunggyung/KoGPT2-FineTuning/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/gyunggyung/KoGPT2-FineTuning/issues)
[![GitHub issues](https://img.shields.io/github/issues/gyunggyung/KoGPT2-FineTuning?style=flat&color=yellow)](https://github.com/gyunggyung/KoGPT2-FineTuning/issues)
[![GitHub stars](https://img.shields.io/github/stars/gyunggyung/KoGPT2-FineTuning?style=social)](https://github.com/gyunggyung/KoGPT2-FineTuning)


SKT-AI에서 약 20GB의 한국어 데이터를 Pre-Training 시킨 [KoGPT2](https://github.com/SKT-AI/KoGPT2)를 사용했습니다. 첫 번째로 가사 작사를 위해서, 정제된 한국어 가사 데이터 486MB를 Fine-tuning 한 결과물입니다. 아래에서, 다양한 한국어 가사를 학습한 결과를 확인 할 수 있습니다. 우리는 이외에도 다양한 프로젝트를 진행할 것입니다.

## Sample
![](img/outputs.JPG)  

## Fine Tuning
```
python main.py --epoch=200 --data_file_path=./dataset/All_make_lyrics_dataset.txt --save_path=./checkpoint/ --load_path=./checkpoint/auto_enter/KoGPT2_checkpoint_18500.tar --batch_size=8
```

### parser
``` python
parser.add_argument('--epoch', type=int, default=200,
					help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='./checkpoint/',
					help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default='./checkpoint/Alls/KoGPT2_checkpoint_296000.tar', 
					help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--samples', type=str, default="samples/",
					help="생성 결과를 저장할 경로입니다.")
parser.add_argument('--data_file_path', type=str, default='dataset/lyrics_dataset.txt',
					help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=8,
					help="batch_size 를 지정합니다.")
```

### Use Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x49fRFi-pgW_P8_Av5fCyYVPvT_9btai)

Colab을 이용해서 Fine-tuning Code를 실행할 수 있습니다.  

![](img/colab_main_img.JPG)

#### Runtime Disconnection Prevention
``` javascript
function ClickConnect() {
    // 백엔드를 할당하지 못했습니다.
    // GPU이(가) 있는 백엔드를 사용할 수 없습니다. 가속기가 없는 런타임을 사용하시겠습니까?
    // 취소 버튼을 찾아서 클릭
    var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); 
    buttons.forEach(function(btn) {
		btn.click();
    });
    console.log("1분 마다 다시 연결");
    document.querySelector("#top-toolbar > colab-connect-button").click();
}
setInterval(ClickConnect,1000*60);
```

#### Clear the screen every 10 minutes
``` javascript
function CleanCurrentOutput(){ 
	var btn = document.querySelector(".output-icon.clear_outputs_enabled.output-icon-selected[title$='현재 실행 중...'] iron-icon[command=clear-focused-or-selected-outputs]");
	if(btn) {
		console.log("10분 마다 출력 지우기");
		btn.click();
	}
} 
setInterval(CleanCurrentOutput,1000*60*10);
```

### GPU Memory Check
```
nvidia-smi.exe
```
## generator
```
python generator.py --temperature=1.0 --text_size=1000 --tmp_sent=""
```
#### 표절 없음
```
python generator.py --temperature=5.0 --text_size=500 --tmp_sent=""
```

### parser
``` python
parser.add_argument('--temperature', type=float, default=0.7,
					help="temperature 를 통해서 글의 창의성을 조절합니다.")
parser.add_argument('--top_p', type=float, default=0.9,
					help="top_p 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--top_k', type=int, default=40,
					help="top_k 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--text_size', type=int, default=250,
					help="결과물의 길이를 조정합니다.")
parser.add_argument('--loops', type=int, default=-1,
					help="글을 몇 번 반복할지 지정합니다. -1은 무한반복입니다.")
parser.add_argument('--tmp_sent', type=str, default="사랑",
					help="글의 시작 문장입니다.")
parser.add_argument('--load_path', type=str, default="./checkpoint/Alls/KoGPT2_checkpoint_296000.tar",
					help="학습된 결과물을 저장하는 경로입니다.")
```

### Use Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qfL-IUp4k0uzkr_6SIaAmS_PA_Luvt1t)

Colab을 이용해서 generator를 실행할 수 있습니다.  

![](img/colab_generator.JPG)

## tensorboard
학습에 따른 변화를 확인하기 위해서, [tensorboard](http://localhost:6006/)로 접근하여 loss와 text를 확인합니다.

```
tensorboard --logdir=runs
```

### loss
![](img/tensorboard_avg.JPG)  

### text
![](img/tensorboard_text.JPG)

## Citation
```
@misc{KoGPT2-FineTuning,
  author = {gyung},
  title = {KoGPT2-FineTuning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gyunggyung/KoGPT2-FineTuning}},
}
```

## Output
자세한 결과물은 [samples](https://github.com/gyunggyung/KoGPT2-FineTuning/tree/master/samples)에서 확인 할 수 있습니다. 학습에 대해서는 [관련 포스팅](https://hipgyung.tistory.com/110)에서 확인할 수 있습니다.

## Reference
> https://github.com/openai/gpt-2  
> https://github.com/nshepperd/gpt-2  
> https://github.com/SKT-AI/KoGPT2  
> https://github.com/asyml/texar-pytorch/tree/master/examples/gpt-2  
> https://github.com/graykode/gpt-2-Pytorch  
> https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317  
> https://github.com/shbictai/narrativeKoGPT2  
> https://github.com/ssut/py-hanspell  
> https://github.com/likejazz/korean-sentence-splitter  
