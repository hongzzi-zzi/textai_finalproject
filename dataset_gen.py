#%%
import pandas as pd
import random
from transformers import ElectraTokenizer, ElectraForTokenClassification
import torch
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ElectraForTokenClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=len(labels)).to(device)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        
#%%

# CSV 파일을 pandas DataFrame으로 읽어옵니다.
df = pd.read_csv('/home/hongeun/23-2/text_ai/final/textai_finalproject/piifin.csv')  # 주어진 CSV 파일의 경로를 사용하세요.

# 템플릿 문장 리스트를 만듭니다.
sns_templates = [
    "안녕 {name}! 오늘 어때?",
    "{name}님, 무슨 일 있어?",
    "{name}야, 이메일은 {mail}이지!",
    "헤이 {name}, 주민등록번호는 {ssn} 맞지?",
    "안녕 {name}! 생일 축하해 🎉 {age} 살이 되었네!",
    "{name}님, 어디 사시는데? {address}?",
    "{name}, {company} 다니면서 {job} 하시나봐!",
    "연락처는 {phone_number}로 주시면 돼!",
    "{name}님, 생일 {birthdate} 맞아? 축하해요!",
    "안녕하세요, {company}에서 일하시는 {name}님!",
    "{name}, {sex}이시군요! 반가워요!",
    "안녕 {name}! {job} 멋지게 하고 있어요.",
    "{name}씨, {company}에서 일하고 계시면 너무 멋져요!",
    "헤이 {name}! 어떤 취미 있어?",
    "{name}, 주민등록번호 {ssn}으로 확인했어!",
    "오늘 {company}에서 첫날을 보냈어. {job}으로 시작하는 새로운 도전이지만, 이미 팀원들과 잘 어울리고 있어. 새로운 환경에서 성장하고 싶어. 연락은 {phone_number}로 해주세요!",
    "생일 {birthdate}이 지나고 나니 이제 {age}살이 되었어. 생일을 맞이한 기분이 아직도 실감나지 않아. 친구들과 가족 덕분에 행복한 하루였어. 모두에게 감사해!",
    "{address}로 이사 온 지 한 달. 새로운 집, 새로운 이웃과의 만남이 즐거워. 이곳에서의 생활이 기대돼. 새로운 시작을 위해 모두들 응원해줘!",
    "오늘 {birthdate}에 {age}살이 되었어! 생일을 맞이해서 가족들과 소중한 시간을 보냈어. 감사하고 사랑해, 모두!",
    "이번 주 {company}에서 {job}으로 새로운 출발을 했어. 첫 주부터 바쁘게 시작했지만, 새로운 도전에 설레고 있어. 앞으로 좋은 결과를 위해 노력할게!",
    "{address}에서의 새로운 생활이 시작된 지 벌써 한 달이 되었네. 이곳에서 새로운 추억을 많이 만들고 싶어. 모두들 종종 놀러 와요!",
    "오늘 {birthdate}에 {age}살이 되었어. 친구들과 함께한 생일파티가 너무 재미있었어. 새로운 나이에 새로운 도전을 기대하고 있어!",
    "{company}에서 {job}으로 첫 주를 보냈어. 새로운 직장 생활에 기대 반, 긴장 반. 앞으로 열심히 할게, 모두 응원해줘!",
    "{birthdate} 생일이 지나고 나니 이제 {age}살이 되었어. 새로운 나이에 새로운 시작을 약속하며, 올해는 더 많은 성장을 이루고 싶어.",
    "이사한 지 한 달, {address}에서의 생활이 너무 좋아. 새로운 환경에서의 매일매일이 즐거워. 여기서 새로운 시작을 할 생각에 벌써부터 설레고 있어.",
    "오늘 {birthdate}로 {age}살이 되었어! {birthdate} 생일을 맞이해서 친구들과 잊지 못할 파티를 열었어. 이제 시작된 새로운 나이에 걸맞게 더 많은 도전과 모험을 기대하고 있어. 모든 친구들과 가족에게 감사의 마음을 전하고 싶어!",
    "이사 완료! 이제 {address}가 내 새로운 집이야. 이사 첫날부터 정신없이 바빴지만, 이제서야 정착한 기분이 들어. 새로운 환경에서 새로운 시작을 할 생각에 설레고 있어. 여러분, 새 집에 놀러 오세요!",
    "오늘부터 {company}에서 {job}로 일하기 시작했어. 새로운 직장에서 첫날은 언제나 떨리고 긴장되지만, 동시에 새로운 시작에 대한 기대도 커. 새로운 동료들과 함께 할 많은 프로젝트들이 기다리고 있어. 여러분의 응원과 격려가 큰 힘이 될 거야!",
    "생일이 지나고 나니 {age}살이 되었네. {birthdate}에 태어난 것을 자축하며, 이제 새로운 한 해를 시작해. 지난 한 해 동안 많은 것을 배우고 경험했는데, 이번 해에도 더 많은 성장과 발전을 기대하고 있어.",
    "안녕하세요, {name}입니다. 최근에 {company}에서 {job}으로 일하게 되었어요. 새로운 업무와 새로운 환경에서 많은 것을 배울 수 있을 것 같아 설레요. 앞으로 좋은 성과를 내기 위해 최선을 다할 테니, 여러분의 응원 부탁드려요!",
    "드디어 {address}에 정착했어. 새로운 동네, 새로운 이웃들과의 만남이 기대되고, 여기서 시작될 새로운 이야기들이 벌써부터 흥미진진해. 새로운 집에서 많은 추억을 만들고 싶어.",
    "내 생일 {birthdate}을 맞이해서 멋진 하루를 보냈어. 친구들과 함께한 시간이 너무 즐거웠고, 이제 {age}살이 된 것을 실감하고 있어. 새로운 나이에 새로운 목표와 꿈을 가지고 열심히 살아갈 거야.",
    "{company}에서 {job}으로 새로운 출발을 했어. 첫날부터 바쁘게 시작했지만, 새로운 역할과 책임이 많지만, 열심히 해내서 좋은 결과를 만들고 싶어. 앞으로의 성공을 위해 모두들 응원해주세요!",
    "이사한 지 벌써 한 달이 지났네. {address}에서의 생활이 정말 마음에 들어. 새로운 이웃들도 친절하고, 여기서의 생활이 매일 즐거워. 새로운 환경에서 시작하는 새로운 삶에 대한 기대감이 커지고 있어.",
    "드디어 새 신용카드 받았어! 카드번호는 {card_number}. 이제 쇼핑할 준비 완료!",
    "계좌번호 바꿨어요, 새 계좌는 {accoutn_number}. 송금해줄 때 이 번호로 보내주세요!"
]
augmented_templates = [
    "{name}님, {birthdate}에 태어나셨군요! 이메일 주소가 {mail} 맞나요?",
    "안녕하세요, {name}입니다. 제 전화번호는 {phone_number}이고, 주소는 {address}입니다.",
    "{name}, 오늘 {company}에서 어떤 일 있었어? {job} 직무 어때?",
    "만나서 반가워요, {name}! {birthdate} 생일인가요? {age}살이시네요!",
    "{name}씨, {address}에서는 어떻게 지내세요? 주민등록번호가 {ssn}인 것 맞죠?",
    "안녕, {name}! {company}에서 {job}으로 일하는 건 어떤가요?",
    "{name}님의 생일이 {birthdate}라고 들었어요. 정말 {age}살이신가요?",
    "어이, {name}! {phone_number}로 전화해도 될까?",
    "기억해, {name}, {company}에서 {job} 직무를 맡고 있어.",
    "{name}씨, {address}에 살고 계시죠? {company}에서 일하는 것 잘 알고 있어요."
]

# 이 템플릿들을 기존 'sns_templates' 리스트에 추가합니다.
sns_templates.extend(augmented_templates)
#%%
tokenized_nonfilled = [tokenizer.tokenize(sentence) for sentence in sns_templates]

#%%
tagged_sentences = []

for sentence in tokenized_nonfilled:
    inside_pii = False  # PII 데이터 안에 있는지 여부를 나타내는 플래그
    tags = []

    for token in sentence:
        if token == '{':
            inside_pii = True
            continue  # '{' 토큰은 태깅하지 않음
        elif token == '}':
            inside_pii = False
            continue  # '}' 토큰은 태깅하지 않음
        elif inside_pii:
            tags.append('P')  # PII 데이터 안의 토큰에는 'P' 태그를 부여
        else:
            tags.append('NP')  # 그 외의 토큰에는 'NP' 태그를 부여
    
    tagged_sentences.append(tags)

#%%
tokenized_df = pd.DataFrame()

for column in df.columns:
    tokenized_df[column] = df[column].apply(lambda x: tokenizer.tokenize(str(x)))


# %%
#%%
# 모든 문장과 데이터셋 행의 조합에 대한 토큰화된 문장과 태그를 저장할 리스트
all_tokenized_sentences = []
all_tags = []

# 각 행에 대해 모든 템플릿을 적용하여 토큰화하고 태깅합니다.
for index, row in df.iterrows():
    for template in sns_templates:
        # 템플릿 내 PII 위치를 실제 데이터로 대체합니다.
        filled_template = template.format(**row.to_dict())
        # 대체된 템플릿을 토큰화합니다.
        tokenized_sentence = tokenizer.tokenize(filled_template)

        # 태그 리스트를 생성합니다.
        tags = []
        for token in tokenized_sentence:
            tag = 'NP'  # 기본 태그
            for key, value in row.to_dict().items():
                # PII 데이터를 토큰화하고 현재 토큰과 일치하는지 확인합니다.
                if token in tokenizer.tokenize(str(value)):
                    tag = 'P'
                    break
            tags.append(tag)

        # 결과를 저장합니다.
        # all_tokenized_sentences.append(tokenized_sentence)
        all_tokenized_sentences.append(tokenizer.convert_tokens_to_string(tokenized_sentence))
        all_tags.append(tags)


#%%
combined_csv_lines = []

for tokens, tags in zip(all_tokenized_sentences, all_tags):
    # 토큰들과 태그들을 각각 ';'로 구분하여 하나의 문자열로 결합합니다.
    token_line = tokens
    tag_line = " ".join(tags)
    # 토큰 문자열과 태그 문자열을 ';'로 구분하여 추가합니다.
    combined_csv_lines.append(token_line + "\t" + tag_line)

# %%
import random

# combined_csv_lines를 랜덤으로 섞습니다.
random.shuffle(combined_csv_lines)

# 전체 데이터의 길이를 계산합니다.
total_length = len(combined_csv_lines)

# 훈련 데이터와 평가 데이터의 비율을 8:2로 설정합니다.
train_size = int(total_length * 0.8)
eval_size = total_length - train_size

# 훈련 데이터와 평가 데이터로 분할합니다.
train_data = combined_csv_lines[:train_size]
eval_data = combined_csv_lines[train_size:]

# 훈련 데이터와 평가 데이터를 각각 CSV 파일로 저장합니다.
train_file_path = 'train.csv'
eval_file_path = 'test.csv'

with open(train_file_path, "w", encoding="utf-8") as file:
    for line in train_data:
        file.write(line + "\n")

with open(eval_file_path, "w", encoding="utf-8") as file:
    for line in eval_data:
        file.write(line + "\n")

# train_file_path, eval_file_path
# %%
