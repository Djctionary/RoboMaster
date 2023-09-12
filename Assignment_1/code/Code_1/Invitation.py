# 邀请你的n个同学来参观RoboMaster实验室，输入人数n以及你n个同学的姓名。
# 登记完成n个人后，你突然得知n个人中有一位有事来不了，删除他的名字并再邀请一位。
# 最后把所有来参加的同学的姓名保存在txt文本上。
student_names = []
n = 0
while 1:
    print("RoboMaster实验室参观人员名单")
    print("输入 1 添加名单")
    print("输入 2 删除名单")
    print("输入 3 查看名单")
    print("输入 4 保存名单")
    print("输入 5 退出")
    Number = int(input())
    if Number == 1:
        name = str(input("请输入姓名:"))
        student_names.append(name)
        n = n + 1
        print("添加成功")
    elif Number == 2:
        name = str(input("请输入需删除者姓名:"))
        if name in student_names:
            student_names.remove(name)
            print("删除成功")
            n = n - 1
        else:
            print("查无此人")
    elif Number == 3:
        print(student_names)
    elif Number == 4:
        with open("参观同学名单.txt", "w") as file:
            for name in student_names:
                file.write(name + "\n")
        print("保存成功")
    elif Number == 5:
        print("退出成功")
        break
    else:
        print("请输入1~5")
