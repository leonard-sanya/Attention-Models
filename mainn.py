from SelfAttention import SelfAttention
from CrossAttention import CrossAttention
from MultiHeadAttention import MultiheadAttention

input_1 = "The cat is sleeping on the mat."
input_2 = "Le chat dort sur le tapis."

SelfAttention_model = SelfAttention(input_1,mask=True)
CrossAttention_model = SelfAttention(input_1,input_2)
MultiheadAttention_model = SelfAttention(input_1)

def main():

    print("Select type of attention model")
    user = input("1,2 or 3 for Selff, Cross or Multihead Attentio models respectively  ")
    if user == "1":
        SelfAttention_model.fit()
    elif user == "2":
        CrossAttention_model.fit()
    elif user == "3":
        MultiheadAttention_model.fit()
    else:
        print("Invalid Option")
         
if __name__ == "__main__":
   main()