import math

# ==========================================
# 骨架關鍵點索引
# ==========================================
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

# ==========================================
# 1. 基礎類別：所有動作都繼承這個模板
# ==========================================
class ActionRule:
    def __init__(self, name):
        self.name = name

    def check(self, kp):
        """接收單一人的 keypoints (kp)，回傳 True 或 False"""
        raise NotImplementedError("請在子類別實作此方法")

# ==========================================
# 2. 具體動作類別 (各司其職，互不干擾)
# ==========================================

class BeastRule(ActionRule):
    """
    YAJUSENPAI!!!
    """
    def __init__(self):
        super().__init__("YAJUSENPAI!!!")

    def check(self, kp):
        nose = kp[NOSE]
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        l_wr, r_wr = kp[L_WRIST], kp[R_WRIST]

        # 1. 取得兩肩的動態寬度，作為量測的動態標準
        shoulder_width = abs(l_sh[0] - r_sh[0])
        
        # 避免除以零或關節點重疊的極端錯誤
        if shoulder_width == 0:
            return False

        # 2. 計算手腕與鼻子的實際距離
        dist_l_nose = math.hypot(l_wr[0] - nose[0], l_wr[1] - nose[1])
        dist_r_nose = math.hypot(r_wr[0] - nose[0], r_wr[1] - nose[1])

        # 3. 判斷 A：手腕是否高於肩膀 (在影像中，Y 座標越小代表越高)
        wrists_above = (l_wr[1] < l_sh[1]) and (r_wr[1] < r_sh[1])

        # 4. 判斷 B：手腕是否在兩肩內側 (X 座標介於左肩與右肩之間)
        # 使用 min 和 max 可以無視畫面是否經過鏡像翻轉
        left_bound = min(l_sh[0], r_sh[0])
        right_bound = max(l_sh[0], r_sh[0])
        wrists_inside = (left_bound < l_wr[0] < right_bound) and (left_bound < r_wr[0] < right_bound)

        # 5. 判斷 C：手腕靠近頭部 (距離小於肩寬的 0.8 倍，徹底拔除絕對像素)
        close_to_head = (dist_l_nose < shoulder_width * 0.8) and (dist_r_nose < shoulder_width * 0.8)

        # 必須同時滿足上述三個相對條件，才算成功發動 114514
        return wrists_above and wrists_inside and close_to_head
    
class KillerQueenRule(ActionRule):
    """
    JoJo立：殺手皇后 (Killer Queen)
    特徵：一手高舉過頭，另一手往下且往對側身體交叉
    """
    def __init__(self):
        super().__init__("KILLER QUEEN!")

    def check(self, kp):
        nose = kp[NOSE]
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        l_wr, r_wr = kp[L_WRIST], kp[R_WRIST]

        # 判斷手腕是否交叉過身體 (比較 X 軸距離)
        # 如果左手腕距離「右肩」比「左肩」近，代表左手往右邊交叉了
        left_crossed = abs(l_wr[0] - r_sh[0]) < abs(l_wr[0] - l_sh[0])
        right_crossed = abs(r_wr[0] - l_sh[0]) < abs(r_wr[0] - r_sh[0])

        # 高低判斷 (Y 座標越小代表越高)
        right_up = r_wr[1] < nose[1]
        left_down = l_wr[1] > l_sh[1]

        left_up = l_wr[1] < nose[1]
        right_down = r_wr[1] > r_sh[1]

        # 情況 A：右手高舉，左手放下且交叉
        condition_a = right_up and left_down and left_crossed
        
        # 情況 B：左手高舉，右手放下且交叉 (讓玩家可以自由換邊擺 pose)
        condition_b = left_up and right_down and right_crossed

        return condition_a or condition_b

class TPoseRule(ActionRule):
    def __init__(self):
        super().__init__("T-POSE DOMINANCE!")

    def check(self, kp):
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        l_wr, r_wr = kp[L_WRIST], kp[R_WRIST]
        y_aligned = abs(l_wr[1] - l_sh[1]) < 60 and abs(r_wr[1] - r_sh[1]) < 60
        x_spread = l_wr[0] > l_sh[0] + 50 and r_wr[0] < r_sh[0] - 50
        return y_aligned and x_spread

class DabRule(ActionRule):
    def __init__(self):
        super().__init__("DAB ON EM!")

    def check(self, kp):
        nose = kp[NOSE]
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        l_wr, r_wr = kp[L_WRIST], kp[R_WRIST]

        dist_l_nose = math.hypot(l_wr[0] - nose[0], l_wr[1] - nose[1])
        dist_r_nose = math.hypot(r_wr[0] - nose[0], r_wr[1] - nose[1])

        if (dist_r_nose < 80 and l_wr[1] < l_sh[1] and l_wr[0] > l_sh[0] + 80) or \
           (dist_l_nose < 80 and r_wr[1] < r_sh[1] and r_wr[0] < r_sh[0] - 80):
            return True
        return False

class StretchingRule(ActionRule):
    def __init__(self):
        super().__init__("Stretching")

    def check(self, kp):
        nose, l_wr, r_wr = kp[NOSE], kp[L_WRIST], kp[R_WRIST]
        return (l_wr[1] < nose[1] - 50) or (r_wr[1] < nose[1] - 50)

class DozingRule(ActionRule):
    def __init__(self):
        super().__init__("Dozing")

    def check(self, kp):
        nose = kp[NOSE]
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        shoulder_avg_y = (l_sh[1] + r_sh[1]) / 2
        return nose[1] > shoulder_avg_y - 20

# ==========================================
# 3. 總管：負責管理所有動作與優先級
# ==========================================
class ActionDetector:
    def __init__(self):
        # 註冊你的動作：陣列的「順序」就是判斷的「優先級」！
        self.rules = [
            BeastRule(),
            KillerQueenRule(),
            TPoseRule(),
            DabRule(),
            StretchingRule(),
            DozingRule()
        ]
        # 定義需要檢查信心度的基本關節點
        self.required_points = [NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST]

    def detect(self, keypoints):
        if len(keypoints) == 0:
            return "No Person"

        kp = keypoints[0]

        # 確保所需的關鍵點都有被準確偵測 (> 0.5 置信度)
        if any(kp[pt][2] < 0.5 for pt in self.required_points):
            return "Working / Normal"

        # 依序跑過所有的動作規則，有命中就提早回傳
        for rule in self.rules:
            if rule.check(kp):
                return rule.name

        return "Working / Normal"