# while True:
#     n = int(input("Enter the number\t:"))
#     for i in range(n-1):
#         for j in range(1,n+1):
#             print("* ", end = " ")
#         print("")
    
# def return_AlphabetOnly_Strings(*args):
#   return [x for x in args if x.isalpha()]
# print(return_AlphabetOnly_Strings("Sidharth","Sid123","12345","Bennett University"))

# class MyClass:
#     def __init__(self,value):
#         self.value = value

#     def show(self):
#         print(f"Value is {self.value}")

#     @property
#     def value(self):
#         return self._value

# obj = MyClass(10)
# obj.show()
# string1 = "Bharatdesh"
# string2 = "Rastriyata"
# final_string = string1[:6] + string2[4:]
# print(final_string)  # Output: Bharatiyata

# def letter_frequency(sentence):
#     frequency = {}
#     for letter in sentence:
#         if letter.isalpha():
#             if letter in frequency:
#                 frequency[letter] += 1
#             else:
#                 frequency[letter] = 1
#     return frequency

# user_input = input("Enter a sentence: ")
# print("Letter frequency:", letter_frequency(user_input))

 
a=5
b=6
ans=a&b
print(ans)