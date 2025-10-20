import argparse
import random
import math
from typing import List, Tuple, Set, Union
import re


class Fraction:
    """分数类，支持真分数和带分数的表示及四则运算"""

    def __init__(self, numerator: int, denominator: int = 1):
        """初始化分数

        Args:
            numerator: 分子
            denominator: 分母，默认为1（整数）

        Raises:
            ValueError: 若分母为0则抛出异常
        """
        if denominator == 0:
            raise ValueError("分母不能为0")
        # 统一处理分母符号（分母为负时，分子取反）
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        self.numerator = numerator  # 分子
        self.denominator = denominator  # 分母
        self._simplify()  # 初始化时自动约分

    def _simplify(self) -> None:
        """约分分数（私有方法）"""
        if self.numerator == 0:
            self.denominator = 1  # 0的分母固定为1
            return
        # 计算最大公约数进行约分
        gcd = math.gcd(abs(self.numerator), self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

    def to_mixed_number(self) -> Tuple[int, int, int]:
        """转换为带分数形式

        Returns:
            元组 (整数部分, 分子, 分母)，如3'1/2返回(3,1,2)
        """
        if self.numerator == 0:
            return 0, 0, 1

        integer = abs(self.numerator) // self.denominator  # 整数部分
        remainder = abs(self.numerator) % self.denominator  # 余数（新分子）
        sign = 1 if self.numerator > 0 else -1  # 符号处理

        # 分情况返回（纯分数/整数/带分数）
        if integer == 0:
            return 0, sign * remainder, self.denominator
        elif remainder == 0:
            return sign * integer, 0, 1
        else:
            return sign * integer, remainder, self.denominator

    def __str__(self) -> str:
        """转为字符串表示（带分数/纯分数/整数）"""
        integer, num, den = self.to_mixed_number()
        if integer == 0:
            if num == 0:
                return "0"
            return f"{num}/{den}"  # 纯分数
        elif num == 0:
            return str(integer)  # 整数
        else:
            return f"{integer}'{num}/{den}"  # 带分数

    def __add__(self, other: Union['Fraction', int]) -> 'Fraction':
        """加法运算（支持分数+分数、分数+整数）"""
        if isinstance(other, int):
            other = Fraction(other)  # 整数转为分数
        # 通分计算
        common_denominator = self.denominator * other.denominator
        numerator = (self.numerator * other.denominator +
                     other.numerator * self.denominator)
        return Fraction(numerator, common_denominator)

    def __sub__(self, other: Union['Fraction', int]) -> 'Fraction':
        """减法运算（支持分数-分数、分数-整数）"""
        if isinstance(other, int):
            other = Fraction(other)
        common_denominator = self.denominator * other.denominator
        numerator = (self.numerator * other.denominator -
                     other.numerator * self.denominator)
        result = Fraction(numerator, common_denominator)
        if result.numerator < 0:
            raise ValueError("减法结果不能为负数")  # 限制小学范围内非负结果
        return result

    def __mul__(self, other: Union['Fraction', int]) -> 'Fraction':
        """乘法运算（支持分数×分数、分数×整数）"""
        if isinstance(other, int):
            other = Fraction(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def __truediv__(self, other: Union['Fraction', int]) -> 'Fraction':
        """除法运算（支持分数÷分数、分数÷整数）"""
        if isinstance(other, int):
            other = Fraction(other)
        if other.numerator == 0:
            raise ValueError("除数不能为0")
        # 除以分数等于乘倒数
        numerator = self.numerator * other.denominator
        denominator = self.denominator * other.numerator

        if abs(numerator) >= abs(denominator):
            raise ValueError("除法结果必须为真分数")  # 限制小学范围内真分数结果

        return Fraction(numerator, denominator)

    def __eq__(self, other: Union['Fraction', int]) -> bool:
        """判断相等（支持与整数或分数比较）"""
        if isinstance(other, int):
            return self.numerator == other and self.denominator == 1
        if isinstance(other, Fraction):
            # 交叉相乘判断相等
            return (self.numerator * other.denominator ==
                    other.numerator * self.denominator)
        return False

    def __lt__(self, other: Union['Fraction', int]) -> bool:
        """小于比较（支持与整数或分数）"""
        if isinstance(other, int):
            other = Fraction(other)
        return self.numerator * other.denominator < other.numerator * self.denominator

    def __le__(self, other: Union['Fraction', int]) -> bool:
        """小于等于比较"""
        return self < other or self == other

    @classmethod
    def from_string(cls, s: str) -> 'Fraction':
        """从字符串解析分数（支持带分数、纯分数、整数）

        Args:
            s: 字符串，如"3'1/2"（带分数）、"1/2"（纯分数）、"5"（整数）

        Returns:
            解析后的Fraction实例
        """
        s = s.strip()
        if "'" in s:  # 带分数（如3'1/2）
            parts = s.split("'")
            integer = int(parts[0])
            num, den = map(int, parts[1].split('/'))
            total_num = abs(integer) * den + num  # 转换为假分数分子
            if integer < 0:
                total_num = -total_num
            return cls(total_num, den)
        elif '/' in s:  # 纯分数（如1/2）
            num, den = map(int, s.split('/'))
            return cls(num, den)
        else:  # 整数（如5）
            return cls(int(s))


class ExpressionGenerator:
    """表达式生成器（基础版本，未优化效率）"""

    def __init__(self, range_num: int):
        """初始化生成器

        Args:
            range_num: 数值范围（生成的数字不超过此范围）
        """
        self.range_num = range_num
        self.operators = ['+', '-', '×', '÷']  # 支持的运算符
        self.priority = {'+': 1, '-': 1, '×': 2, '÷': 2}  # 运算符优先级

    def generate_number(self) -> Fraction:
        """生成随机数（自然数或真分数）

        Returns:
            随机Fraction实例（30%概率生成真分数，70%概率生成整数）
        """
        if random.random() < 0.3:  # 30%概率生成真分数
            denominator = random.randint(2, self.range_num - 1)
            numerator = random.randint(1, denominator - 1)  # 分子小于分母（真分数）
            return Fraction(numerator, denominator)
        else:  # 70%概率生成整数
            return Fraction(random.randint(0, self.range_num - 1))

    def generate_expression(self, max_operators: int = 3) -> Tuple[str, Fraction]:
        """生成表达式及结果（控制运算符数量≤3）

        Args:
            max_operators: 最大运算符数量（默认3）

        Returns:
            元组 (表达式字符串, 计算结果Fraction)
        """
        if max_operators == 0:  # 递归终止：生成单个数字
            number = self.generate_number()
            return str(number), number

        # 30%概率生成带括号的表达式（仅当运算符数量>1时）
        use_parentheses = random.random() < 0.3 and max_operators > 1

        if use_parentheses:
            # 拆分运算符到左右子表达式
            left_ops = random.randint(0, max_operators - 1)
            right_ops = max_operators - 1 - left_ops

            left_expr, left_value = self.generate_expression(left_ops)
            right_expr, right_value = self.generate_expression(right_ops)

            op = random.choice(self.operators)
            result: Fraction
            try:
                # 计算结果（根据运算符）
                if op == '+':
                    result = left_value + right_value
                elif op == '-':
                    result = left_value - right_value
                elif op == '×':
                    result = left_value * right_value
                else:  # ÷
                    result = left_value / right_value

                expr = f"({left_expr}) {op} ({right_expr})"  # 带括号表达式
                return expr, result

            except (ValueError, ZeroDivisionError):
                # 运算无效时重新生成
                return self.generate_expression(max_operators)
        else:
            # 50%概率生成乘除（高优先级），50%生成加减（低优先级）
            if max_operators > 0 and random.random() < 0.5:
                op = random.choice(['×', '÷'])
            else:
                op = random.choice(['+', '-'])

            # 拆分运算符到左右子表达式
            left_ops = random.randint(0, max_operators - 1)
            right_ops = max_operators - 1 - left_ops

            # 乘除运算限制子表达式复杂度（最多1个运算符）
            if op in ('×', '÷'):
                left_ops = min(left_ops, 1)
                right_ops = min(right_ops, 1)

            left_expr, left_value = self.generate_expression(left_ops)
            right_expr, right_value = self.generate_expression(right_ops)

            # 计算结果并返回表达式
            try:
                if op == '+':
                    result = left_value + right_value
                elif op == '-':
                    result = left_value - right_value
                elif op == '×':
                    result = left_value * right_value
                else:  # ÷
                    result = left_value / right_value

                expr = f"{left_expr} {op} {right_expr}"
                return expr, result

            except (ValueError, ZeroDivisionError):
                # 运算无效时重新生成
                return self.generate_expression(max_operators)

    def is_duplicate(self, expr1: str, expr2: str) -> bool:
        """检查两个表达式是否重复（考虑交换律，未使用缓存）

        Args:
            expr1: 表达式1
            expr2: 表达式2

        Returns:
            若重复则为True，否则为False
        """
        # 预处理：移除空格、等号，替换运算符
        expr1 = expr1.replace(' ', '').replace('=', '')
        expr2 = expr2.replace(' ', '').replace('=', '')
        expr1 = expr1.replace('×', '*').replace('÷', '/')
        expr2 = expr2.replace('×', '*').replace('÷', '/')

        if expr1 == expr2:
            return True

        # 标准化后比较
        return self._normalize_expression(expr1) == self._normalize_expression(expr2)

    def _normalize_expression(self, expr: str) -> str:
        """标准化表达式（处理交换律，无缓存）"""
        # 无加减乘除时直接返回
        if '+' not in expr and '*' not in expr:
            return expr

        try:
            # 移除外层括号
            if expr.startswith('(') and expr.endswith(')'):
                return self._normalize_expression(expr[1:-1])

            # 寻找最低优先级运算符（拆分点）
            min_priority = float('inf')
            split_pos = -1
            parentheses_count = 0  # 括号计数器（处理嵌套）

            for i, c in enumerate(expr):
                if c == '(':
                    parentheses_count += 1
                elif c == ')':
                    parentheses_count -= 1
                elif parentheses_count == 0 and c in self.priority:
                    # 记录最低优先级运算符位置
                    if self.priority[c] < min_priority:
                        min_priority = self.priority[c]
                        split_pos = i

            if split_pos == -1:
                return expr

            op = expr[split_pos]
            left = expr[:split_pos]
            right = expr[split_pos + 1:]

            # 递归标准化左右子表达式
            left_norm = self._normalize_expression(left)
            right_norm = self._normalize_expression(right)

            # 加法和乘法满足交换律，按字典序排序左右部分
            if op in ('+', '*'):
                return f"{min(left_norm, right_norm)}{op}{max(left_norm, right_norm)}"
            else:  # 减法和除法不满足交换律，保持顺序
                return f"{left_norm}{op}{right_norm}"
        except (IndexError, ValueError):
            return expr


class MathExerciseGenerator:
    """数学题目生成器（基础版本，去重效率较低）"""

    def __init__(self, range_num: int):
        """初始化题目生成器

        Args:
            range_num: 数值范围
        """
        self.generator = ExpressionGenerator(range_num)
        self.generated_expressions: Set[str] = set()  # 存储已生成的表达式（原始字符串）

    def generate_exercises(self, count: int) -> List[Tuple[str, str]]:
        """生成指定数量的题目及答案（未优化去重效率）

        Args:
            count: 题目数量

        Returns:
            题目列表，每个元素为(题目字符串, 答案字符串)
        """
        exercises = []
        attempts = 0
        max_attempts = count * 50  # 最大尝试次数

        while len(exercises) < count and attempts < max_attempts:
            attempts += 1
            try:
                # 生成表达式和结果
                expr, result = self.generator.generate_expression()
                exercise = f"{expr} = "

                # 验证结果正确性（双重计算校验）
                evaluated_result = AnswerChecker.evaluate_expression(expr)
                if str(result) != evaluated_result:
                    continue

                # 检查重复（遍历集合对比，效率低）
                is_duplicate = False
                for existing in self.generated_expressions:
                    if self.generator.is_duplicate(exercise, existing):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    self.generated_expressions.add(exercise)
                    exercises.append((exercise, str(result)))
            except (ValueError, ZeroDivisionError):
                continue  # 跳过无效表达式

        if len(exercises) < count:
            print(f"警告：只生成了 {len(exercises)} 个题目，无法生成 {count} 个不重复的合法题目")

        return exercises


class AnswerChecker:
    """答案检查器（验证学生答案与标准答案是否一致）"""

    @staticmethod
    def check_answers(exercise_file: str, answer_file: str) -> Tuple[List[int], List[int]]:
        """检查答案并返回正确/错误题目编号

        Args:
            exercise_file: 题目文件路径
            answer_file: 答案文件路径

        Returns:
            元组 (正确题目编号列表, 错误题目编号列表)

        Raises:
            FileNotFoundError: 若文件不存在
            ValueError: 若题目与答案数量不一致
        """
        try:
            with open(exercise_file, 'r', encoding='utf-8') as f:
                exercises = [line.strip() for line in f.readlines() if line.strip()]

            with open(answer_file, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件未找到: {e}")

        if len(exercises) != len(answers):
            raise ValueError(f"题目文件和答案文件行数不一致：题目 {len(exercises)} 题，答案 {len(answers)} 个")

        correct_indices = []
        wrong_indices = []

        for i, (exercise, student_answer) in enumerate(zip(exercises, answers), 1):
            try:
                # 提取表达式并计算标准答案
                expr = exercise.replace('=', '').strip()
                standard_answer = AnswerChecker.evaluate_expression(expr)

                # 比较答案
                if AnswerChecker._compare_answers(standard_answer, student_answer):
                    correct_indices.append(i)
                else:
                    wrong_indices.append(i)
            except (ValueError, TypeError) as e:
                print(f"处理题目 {i} 时出错: {e}")
                wrong_indices.append(i)

        return correct_indices, wrong_indices

    @staticmethod
    def evaluate_expression(expr: str) -> str:
        """计算表达式的值（支持带分数、分数、整数）"""
        expr = expr.replace('×', '*').replace('÷', '/')  # 替换运算符为Python支持的符号
        # 匹配分数格式：带分数(3'1/2)、普通分数(1/2)、整数(5)
        pattern = r'(\d+\'\d+/\d+|\d+/\d+|\d+)'

        def replace_match(match: re.Match) -> str:
            # 用Fraction.from_string解析分数
            return f"Fraction.from_string('{match.group(1)}')"

        # 替换表达式中的分数为Fraction实例创建代码
        safe_expr = re.sub(pattern, replace_match, expr)

        try:
            # 安全执行表达式（限制命名空间）
            result = eval(safe_expr, {"__builtins__": None}, {"Fraction": Fraction})
            if isinstance(result, Fraction):
                return str(result)
            else:
                return str(Fraction(result))
        except (ValueError, ZeroDivisionError, TypeError) as e:
            raise ValueError(f"表达式计算错误: {expr} - {str(e)}")

    @staticmethod
    def _compare_answers(answer1: str, answer2: str) -> bool:
        """比较两个答案是否相等（支持分数、整数）"""
        try:
            frac1 = Fraction.from_string(answer1)
            frac2 = Fraction.from_string(answer2)
            return frac1 == frac2
        except (ValueError, TypeError):
            return False


def main():
    """主函数：解析命令行参数并执行相应操作（生成题目或检查答案）"""
    parser = argparse.ArgumentParser(description='小学四则运算题目生成器')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', type=int, help='生成题目的数量')
    group.add_argument('-e', type=str, help='题目文件路径')
    parser.add_argument('-r', type=int, help='数值范围')
    parser.add_argument('-a', type=str, help='答案文件路径')

    args = parser.parse_args()

    if args.n is not None:  # 生成题目模式
        if args.r is None:
            parser.error("使用 -n 参数时必须同时使用 -r 参数")
        if args.r < 1:
            parser.error("数值范围 -r 必须是正整数")
        if args.n < 1:
            parser.error("题目数量 -n 必须是正整数")

        generator = MathExerciseGenerator(args.r)
        exercises = generator.generate_exercises(args.n)

        # 保存题目和答案到文件
        with open('Exercises.txt', 'w', encoding='utf-8') as f:
            for exercise, _ in exercises:
                f.write(exercise + '\n')

        with open('Answers.txt', 'w', encoding='utf-8') as f:
            for _, answer in exercises:
                f.write(answer + '\n')

        print(f"已生成 {len(exercises)} 个题目到 Exercises.txt")
        print(f"答案已保存到 Answers.txt")

    else:  # 检查答案模式
        if args.e is None or args.a is None:
            parser.error("使用 -e 参数时必须同时使用 -a 参数")

        try:
            correct, wrong = AnswerChecker.check_answers(args.e, args.a)

            with open('Grade.txt', 'w', encoding='utf-8') as f:
                f.write(f"Correct: {len(correct)} {tuple(correct)}\n")
                f.write(f"Wrong: {len(wrong)} {tuple(wrong)}\n")

            print(f"检查完成！结果已保存到 Grade.txt")
            print(f"正确: {len(correct)} 题")
            print(f"错误: {len(wrong)} 题")

        except Exception as e:
            print(f"错误: {e}")


if __name__ == '__main__':
    main()
