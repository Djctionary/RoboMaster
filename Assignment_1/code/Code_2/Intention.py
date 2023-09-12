class Student:
    def __init__(self, name):
        self.name = name
        self.desire = {
            'electric_control': False,
            'mechanical': False,
            'hardware': False,
            'vision': False,
            'operation': False
        }
        self.vision_direction = "False"

    def choose_group(self, group):
        self.desire[group] = True

    def choose_vision_direction(self, direction):
        self.vision_direction = direction

    def to_string(self):
        return f"Name: {self.name}\nDesire: {self.desire}\nVision direction: {self.vision_direction}"


n = int(input("请输入参观的同学人数："))
students = []

for i in range(n):
    name = input(f"请输入第{i+1}个同学的姓名：")
    student = Student(name)
    group = int(input("请选择你要参加的组别（请输入数字）：\n1. electric_control \
    \n2. mechanical\n3. hardware\n4. vision\n5. operation\n"))
    if group == 4:
        student.choose_group("vision")
        direction = int(input("请选择你想去的视觉方向（请输入数字）：\n1. OpenCV\n2. DL \
                              \n3. SLAM\n"))
        if direction == 1:
            student.choose_vision_direction("OpenCV")
        elif direction == 2:
            student.choose_vision_direction("DL")
        elif direction == 3:
            student.choose_vision_direction("SLAM")
        students.append(student)

    elif group == 1:
        student.choose_group("electric_control")
        students.append(student)
    elif group == 2:
        student.choose_group("mechanical")
        students.append(student)
    elif group == 3:
        student.choose_group("hardware")
        students.append(student)
    elif group == 5:
        student.choose_group("operation")
        students.append(student)


with open('students.txt', 'w') as f:
    for student in students:
        f.write(student.to_string() + "\n")