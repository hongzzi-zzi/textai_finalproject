#%%
# Faker 라이브러리를 사용하여 다양한 PII 정보를 포함하는 템플릿 생성
from faker import Faker
import pandas as pd
import string
from datetime import datetime
import time
import subprocess
from faker.providers import BaseProvider
import random
import re
import pickle

import subprocess




class CustomProvider(BaseProvider):
    def korean_phone_number(self):
        return f"010-{random.randint(0, 9999):04d}-{random.randint(0, 9999):04d}"
    
    def korean_email(self):
        korean_email_domains = ['gmail.com', 'naver.com', 'daum.net', 'hanmail.net', 'kakao.com']

        # 랜덤으로 이메일 도메인을 선택합니다.
        selected_domain = random.choice(korean_email_domains)
        pattern = r'\bexample\..*\b'

        # 정규 표현식을 사용하여 패턴을 찾아서 대체합니다.
        return re.sub(pattern, selected_domain, fake.email())
    
    def korean_card(self):
        return f"{random.randint(0, 9999):04d}-{random.randint(0000, 9999):04d}-{random.randint(1000, 9999):04d}-{random.randint(1000, 9999):04d}"
    
    def korean_acc_num(self):
        banks = ['kakao', 'toss', 'sinhan', 'hana', 'kukmin']
        selected_bank = random.choice(banks)

        if selected_bank == 'kakao':
            account_number = f'3333-{random.randint(0, 99):02d}-{random.randint(0, 999999999):07d}'
        elif selected_bank == 'toss':
            account_number = f'{random.randint(0, 9999):04d}-{random.randint(0, 9999):04d}-{random.randint(0, 9999):04d}'
        elif selected_bank == 'sinhan':
            account_number = f'110-{random.randint(0, 999):03d}-{random.randint(0, 999999):06d}'
        elif selected_bank == 'hana':
            account_number = f'{random.randint(0, 999):03d}-{random.randint(0, 999999):06d}-{random.randint(0, 999):03d}08'
        elif selected_bank == 'kukmin':
            account_number = f'{random.randint(0, 9999):04d}24-{random.randint(0, 99):02d}-{random.randint(0, 999999):06d}'

        return account_number



# Faker 인스턴스 초기화 (한국어 설정)
fake = Faker('ko_KR')
fake.add_provider(CustomProvider)



def create_consistent_ssn(birthdate, sex):
    # birthdate는 'yyyy-mm-dd' 형식의 문자열이라고 가정
    year, month, day = str(birthdate).split('-')
    
    # 2000년 이전과 이후를 구분하여 성별 구분 번호 설정
    if int(year) < 2000:
        gender_code = '2' if sex == 'female' else '1'
    else:
        gender_code = '4' if sex == 'female' else '3'

    # 뒷자리의 나머지 숫자 생성 (랜덤하게 6자리 숫자)
    remaining = f"{random.randint(0, 999999):06d}"

    # 앞자리는 태어난 연도의 마지막 두 자리와 월, 일을 사용
    birth_year_short = year[-2:]

    # 완성된 주민등록번호 반환
    ssn = f"{birth_year_short}{month}{day}-{gender_code}{remaining}"
    return ssn

def create_age(birthdate):
    # birthdate가 datetime.date 타입이라면 문자열 파싱이 필요 없음
    today = datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age


with open('/home/hongeun/23-2/text_ai/final/address.pkl', 'rb') as file:
    address = pickle.load(file).tolist()
    
company_df = pd.read_csv("/home/hongeun/23-2/text_ai/final/company.csv")
company_names = company_df["회사명"]
    
def create_random_profile():
    # print('------------------------------------------')
    birthdate=fake.date_of_birth()
    sex=fake.random_element(elements=('male', 'female'))
    
    return {
        'name':fake.name(),
        'age': create_age(birthdate), 
        'sex': sex,
        'mail':fake.korean_email(),
        'ssn': create_consistent_ssn(birthdate, sex),
        'address':random.choice(address),
        'job': fake.job(),
        'phone_number': fake.korean_phone_number(),
        'card_number': fake.korean_card(),
        'accoutn_number': fake.korean_acc_num(),
        'birthdate': birthdate,
        'company': random.choice(company_names),
    }

# %%
profile_list = []

# profile = create_random_profile()
#%%
# 100번 반복하여 프로필 생성하고 리스트에 추가
for _ in range(100):
    profile = create_random_profile()
    print(profile)
    profile_list.append(profile)

#%%

# 리스트를 DataFrame으로 변환
df = pd.DataFrame(profile_list, index=None)

# DataFrame을 CSV 파일로 저장
df.to_csv('piifin.csv', index=False)
# %%
