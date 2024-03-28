from nltk.tokenize import word_tokenize

#read the question and answer data from the file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def getMarks(question):
    s = question[(question.rindex('[')+1):(question.rindex(']'))]
    return float(s)

if __name__ == '__main__':
    question = read_data('Data\\Q1\\question.txt')[0]
    marks = getMarks(question)
    question = question[:(question.rindex('['))]
    model = read_data('Data\\Q1\\model.txt')[0].lower()
    
    file = open("Data\\Q1\\dataset.csv","a")
    for i in range(1,4):
        answer = read_data('Data\\Q1\\answer'+str(i)+'.txt')[0].lower()
        file.write("\n" + question + "," + answer + "," + marks + ",")
        
        