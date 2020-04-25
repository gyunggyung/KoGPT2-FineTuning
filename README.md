# KoGPT2-FineTuning
SKT-AI에서 약 20GB의 한국어 데이터를 Pre-Training 시킨 [KoGPT2](https://github.com/SKT-AI/KoGPT2)를 사용했습니다. 첫 번째로 가사 작사를 위해서, 정제된 한국어 가사 데이터 62MB를 Fine-tuning 한 결과물입니다. 다양한 한국어 가사를 학습한 결과를 확인 할 수 있습니다. checkpoint는 [구글 드라이브](https://drive.google.com/drive/folders/18CRYESHHE897CIaodZj0m96tAI6Vk5wX)에서 확인 할 수 있습니다.

## Fine Tuning
```
python main.py
```

## generator
```
python generator.py --temperature=0.9 --input_size=500 --tmp_sent="가자" --loops=5
```

### parser
``` python
	parser.add_argument('--temperature', type=float, default=0.7, help="temperature 를 통해서 글의 창의성을 조절합니다.")
	parser.add_argument('--top_p', type=float, default=0.9, help="top_p 를 통해서 글의 창의성을 조절합니다.")
	parser.add_argument('--top_k', type=int, default=40, help="top_k 를 통해서 글의 창의성을 조절합니다.")
	parser.add_argument('--input_size', type=int, default=250, help="글의 길이를 조정합니다.")
	parser.add_argument('--loops', type=int, default=-1, help="글을 몇 번 반복할지 지정합니다. -1은 무한반복입니다.")
	parser.add_argument('--tmp_sent', type=str, default="사랑", help="글의 시작 문장입니다.")
```

## Output
더 자세한 결과물은 [samples](https://github.com/gyunggyung/KoGPT2-FineTuning/tree/master/samples)에서 확인 할 수 있습니다.

### "사랑"
```
사랑이란 아프고 아픈 것 yeah 이별이란 아프고 더 아픈 것 같애 니가 없으면 나 안될 것 같아 사랑해줘 사랑해줘 다시 내 품으로 와줘 사랑이란 아프고 아픈 것 yeah 이별이란 아프고 더 아픈 것 같애 니가 없으면 나 안될 것 같아 사랑해줘 사랑해줘 다시 내 품으로 와줘 <|endoftext|> 
```

```

```

```

```

```

```

### " "
```
[Hook: V] 왜 내 맘을 흔드는 건데 왜 내 맘을 흔드는 건데 흔드는 건데 흔드는 건데 [Verse 2: RM] 아빠, 아<unk> Hangul 아무리 애를 써봐도 널 볼 수 없잖아 또 눈물이 나잖아 널 잊어야 하는데 [Verse 3: Jungkook] 사랑이란 아프고 아픈 것 yeah 이별이란 아프고 더 아픈 것 같애 니가 없으면 나 안될 것 같아 사랑해줘 사랑해줘 다시 내 품으로 와줘 [Verse 4: Jimin & V] 시린 널 불어내 본다 연기처럼 하얀 연기처럼 말로는 지운다 해도 사실 난 아직 널 보내지 못하는데 [Pre-Chorus 1: Jungkook & V] 눈꽃이 떨어져요 또 조금씩 멀어져요 보고 싶다 (보고 싶다</s>
```

```
너는 나의 사랑의 배터리 [Verse 3: Young B] 네가 떠나 버린 후 내 가슴은 텅 비었어 내가 너를 놓쳤어 넌 나의 배터리 <|endoftext|> [Verse 1: Jennie] Baby 날 터질 것처럼 안아줘 그만 생각해 뭐가 그리 어려워 거짓말처럼 키스해줘 내가 너에게 마지막 사랑인 것처럼 [Hook: Lisa] 마지막처럼 마-마-마지막처럼 마지막 밤인 것처럼 love 마지막처럼 마-마-마지막처럼 내일 따윈 없는 것처럼 [Verse 2: Lisa] Uh, I'ma fall in love, baby You gon' finna catch me Uh, give you
```

```

```

```

```

### "<|endoftext|>"
```

```

```

```

```

```

```

```

### "Other"

```
가만 보니 네가 내 옆에 있어 줄 때 난 다시 잠이 들 것 같아 [Verse 2: Jin] 네가 내게 말했지, "넌 날 잊었어" 그때가 내 기억 속에 선명해 넌 날 잊었어 [Pre-Chorus: Jungkook, Jimin] 한때는 태양의 세계에 속했던 노랜 멈췄어 노랜
```

```
미친 세상 속에 너는 날 구원해 [Chorus: Jungkook, Jimin] 넌 나의 구원 넌 나의 창 난 너만 있으면 돼 You know that I can't Show you me Give you me 초라한 모습 보여줄 순 없어 또 가면을 쓰고 널 만나러 가 But I still want you [Post-Chorus: Jin] 어쩌면 그때 조금만 이만큼만 용길 내서 너의 앞에 섰더라면 지금 모든 건 달라졌어
```

```
분노와 외로움 그 무엇도 남지 않게끔 [Chorus: Jungkook, Jimin] I'm so sick of this fake love, fake love, fake love I'm so sorry but it's fake love, fake love, fake love [Verse 2: RM] I wanna be a good man just for you 세상을 줬네 just for you 전부 바꿨어
```

```
[Verse 1: GIRIBOY] I'm a born hater. 달리, 반, 피카소? 난 벨라스케스, 밀레, 엘 fuckin' 그레코 내 에코. VJ의 감성 shit? 다 보급형 블로. 내 아류, 문하생 shit 내 원래 성격은 이렇게 나와 문제 하나 없어도 fuck 'em So I
```

```

```

```

```

```

```

```

```