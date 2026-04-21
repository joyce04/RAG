from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

def main():
    print("Adaptive-rag for Korean Competition Law Cases")
    print(app.invoke(input={'question': '플랫폼 업체의 시장지배적지위의 남용행위에 대한 판례를 찾아줘'}))


if __name__ == "__main__":
    main()
