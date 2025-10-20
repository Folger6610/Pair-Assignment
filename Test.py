import unittest  # 单元测试框架
import os        # 文件操作
import tempfile  # 临时文件
import time      # 计时（测试性能）
# 导入待测试的模块和类
from Math import MathExerciseGenerator, AnswerChecker, Fraction


class TestMathExerciseGenerator(unittest.TestCase):
    """测试题目生成器核心功能（涵盖生成、去重、运算合法性等）"""

    def setUp(self):
        """测试前初始化：创建题目生成器实例（范围10以内）"""
        self.test_range = 10  # 测试用数值范围
        self.generator = MathExerciseGenerator(self.test_range)

    def test_basic_generation(self):
        """测试基础题目生成功能（验证数量和格式）"""
        exercises = self.generator.generate_exercises(10)
        self.assertEqual(len(exercises), 10, "应生成10道题目")  # 数量校验
        for expr, ans in exercises:
            self.assertTrue(expr.endswith(" = "), "题目格式应为'表达式 = '")  # 格式校验
            self.assertTrue(len(ans) > 0, "答案不能为空")  # 答案非空校验

    def test_large_scale_generation(self):
        """测试大规模题目生成性能（1000道题应在10秒内完成）"""
        start_time = time.time()
        exercises = self.generator.generate_exercises(1000)
        end_time = time.time()
        self.assertEqual(len(exercises), 1000, "应生成1000道题目")  # 数量校验
        self.assertLess(end_time - start_time, 10, "1000题生成应在10秒内完成")  # 性能校验

    def test_duplicate_expressions(self):
        """测试题目去重功能（确保无重复题目）"""
        exercises = self.generator.generate_exercises(500)
        expr_set = set()  # 存储标准化后的表达式
        for expr, _ in exercises:
            normalized = self.generator.generator.normalize_expression(expr)
            self.assertNotIn(normalized, expr_set, "存在重复题目")  # 去重校验
            expr_set.add(normalized)

    def test_operator_count(self):
        """测试表达式中运算符数量不超过3个"""
        exercises = self.generator.generate_exercises(100)
        operators = ['+', '-', '×', '÷']  # 目标运算符
        for expr, _ in exercises:
            # 移除括号和等号，提取纯表达式
            clean_expr = expr.replace('(', '').replace(')', '').replace(' = ', '')
            # 统计运算符数量
            count = sum(1 for c in clean_expr if c in operators)
            self.assertLessEqual(count, 3, "运算符数量不能超过3个")  # 数量限制校验

    def test_subtraction_non_negative(self):
        """测试减法结果非负（符合小学运算要求）"""
        exercises = self.generator.generate_exercises(100)
        for expr, ans in exercises:
            if '-' in expr:  # 只检查含减法的表达式
                ans_frac = Fraction.from_string(ans)
                self.assertGreaterEqual(ans_frac.numerator, 0, "减法结果不能为负数")  # 非负校验

    def test_division_proper_fraction(self):
        """测试纯除法表达式的结果为真分数（分子<分母）"""
        exercises = self.generator.generate_exercises(100)
        operators = ['+', '-', '×', '÷']

        for expr, ans in exercises:
            # 提取表达式中的运算符
            clean_expr = expr.replace('(', '').replace(')', '').replace(' = ', '')
            op_list = [c for c in clean_expr if c in operators]

            # 只检查：仅包含一个除法运算符，且无其他运算符的表达式
            if op_list == ['÷']:
                ans_frac = Fraction.from_string(ans)
                self.assertLess(
                    abs(ans_frac.numerator),
                    ans_frac.denominator,
                    f"纯除法表达式 {expr} 结果 {ans} 不是真分数（分子{ans_frac.numerator}，分母{ans_frac.denominator}）"
                )

    def test_mixed_number_operation(self):
        """测试带分数运算正确性（手动构造已知结果的案例）"""
        # 测试 3'1/2 + 1'1/3 = 4'5/6（预期结果）
        expr = "3'1/2 + 1'1/3"
        expected_ans = "4'5/6"
        actual_ans = AnswerChecker.evaluate_expression(expr)
        self.assertEqual(actual_ans, expected_ans, "带分数加法计算错误")

    def test_fraction_string_parsing(self):
        """测试分数字符串解析功能（验证各种格式的分数解析是否正确）"""
        # 测试案例：(输入字符串, 预期Fraction实例)
        cases = [
            ("3/4", Fraction(3, 4)),          # 纯分数
            ("2'1/3", Fraction(7, 3)),        # 带分数（2*3+1=7）
            ("-1'1/2", Fraction(-3, 2)),      # 负带分数（-(1*2+1)=-3）
            ("5", Fraction(5, 1))             # 整数
        ]
        for s, expected in cases:
            parsed = Fraction.from_string(s)
            self.assertEqual(parsed, expected, f"解析'{s}'失败")  # 解析结果校验


class TestAnswerChecker(unittest.TestCase):
    """测试答案检查器功能（涵盖答案验证、文件处理等）"""

    def setUp(self):
        """测试前准备：创建临时题目文件和答案文件路径"""
        # 生成临时文件路径（使用id确保唯一性）
        self.exercise_path = os.path.join(tempfile.gettempdir(), f"test_ex_{id(self)}.txt")
        self.answer_path = os.path.join(tempfile.gettempdir(), f"test_ans_{id(self)}.txt")

    def tearDown(self):
        """测试后清理：删除临时文件（容错处理）"""
        for path in [self.exercise_path, self.answer_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass  # 忽略文件占用错误（避免测试中断）

    def _write_files(self, exercises, answers):
        """辅助方法：向临时文件写入题目和答案"""
        with open(self.exercise_path, 'w', encoding='utf-8') as f:
            f.writelines(exercises)
        with open(self.answer_path, 'w', encoding='utf-8') as f:
            f.writelines(answers)

    def test_correct_answers(self):
        """测试正确答案识别（应全部判定为正确）"""
        self._write_files(
            ["1 + 2 = \n", "3 × 4 = \n"],  # 题目
            ["3\n", "12\n"]                # 正确答案
        )
        correct, wrong = AnswerChecker.check_answers(self.exercise_path, self.answer_path)
        self.assertEqual(correct, [1, 2], "应识别所有正确答案")  # 正确答案校验
        self.assertEqual(wrong, [], "不应有错误答案")  # 错误答案校验

    def test_wrong_answers(self):
        """测试错误答案识别（应全部判定为错误）"""
        self._write_files(
            ["5 - 3 = \n", "2 ÷ 4 = \n"],  # 题目
            ["1\n", "1/3\n"]                # 错误答案（正确应为2和1/2）
        )
        correct, wrong = AnswerChecker.check_answers(self.exercise_path, self.answer_path)
        self.assertEqual(correct, [], "不应有正确答案")  # 正确答案校验
        self.assertEqual(wrong, [1, 2], "应识别所有错误答案")  # 错误答案校验

    def test_file_mismatch(self):
        """测试题目与答案数量不一致时的报错（应抛出ValueError）"""
        self._write_files(
            ["1 + 1 = \n"],  # 1道题目
            ["2\n", "3\n"]   # 2个答案（数量不匹配）
        )
        with self.assertRaises(ValueError):
            AnswerChecker.check_answers(self.exercise_path, self.answer_path)  # 异常校验

    def test_parentheses_operation(self):
        """测试带括号表达式的计算正确性（验证括号优先级处理）"""
        expr = "(1/2 + 1/3) × 6"
        expected_ans = "5"  # 计算逻辑：(3/6 + 2/6)×6 = 5/6×6 = 5
        actual_ans = AnswerChecker.evaluate_expression(expr)
        self.assertEqual(actual_ans, expected_ans, "带括号表达式计算错误")  # 结果校验


if __name__ == '__main__':
    unittest.main(verbosity=2)  # 执行所有测试（verbosity=2显示详细信息）
