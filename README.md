
# Simulated Smooth RNN

Simulated Smooth RNN (SS-RNN), a recurrent architecture for sequence processing. Its core innovation is a memory mechanism that achieves flexible, continuous-space addressing at a constant O(1) computational cost. SS-RNN uses a controller for its multi-head read/write system to generate floating-point addresses (e.g., 50.6). These "Smooth" addresses are then "Simulated" by a differentiable linear interpolation operation that interacts only with the two nearest discrete memory slots to execute all memory actions.


### I. High-Level Concept

This is a **recurrent** memory unit，for managing the states produced during sequence processing.

1.  **External Dimension ($n$):** The model's "public interface" dimension (e.g., $n=768$). It receives an $n$-dimensional $x_k$ (input) and produces an $n$-dimensional $y_k$ (output).
2.  **Internal Dimension ($r$):** The model's "core memory" dimension (e.g., $r=64$). $r$ is much smaller than $n$ ($r \ll n$).

All complex, expensive memory operations (read, write, forget) occur in this **lightweight $r$-dimensional space**, achieving extremely high computational and memory efficiency. The model **down-projects** the $n$-dimensional input to $r$-dimensions at the "entrance" and **up-projects** the $r$-dimensional result back to $n$-dimensions at the "exit".

### II. Core Components & Symbol Definitions

* **$k$**: The current timestep.
* **$x_k$**: The $n$-dimensional input embedding vector (e.g., $n=768$).
* **$y_k$**: The $n$-dimensional output vector.
* **$H_k$**: The **Memory Sandbox**. This is an $M \times r$ tensor ($M$ slots, each $r$-dimensional).
* **$M$**: The total number of slots in the memory sandbox (hyperparameter, e.g., $M=1000$).
* **$n$**: The external dimension (e.g., $n=768$).
* **$r$**: The **internal core dimension** (hyperparameter, e.g., $r=64$).
* **$\text{MLP}_{\text{down}}$**: Entrance projection MLP, responsible for $n \to r$ down-projection.
* **$\text{MLP}_{\text{up}}$**: Exit projection MLP, responsible for $r \to n$ up-projection.
* **$\text{MLP}_{\text{R/F/W/G}}$**: Various "controller" MLPs, which now all operate in $r$-dimensional space.
* **$K_r, K_w, K_f$**: The number of read, write, and forget operation "Heads".
* **$t$**: An operation position (scalar), in the range `[0, M-1]`.
* **$v$**: An $r$-dimensional value vector.
* **$g$**: An $r$-dimensional gate vector (values 0-1).


### III. Detailed Single-Step (k-th Timestep) Forward Pass

At timestep $k$, the model receives the $n$-dimensional $x_k$ and the $M \times r$ sandbox state $H_{k-1}$.

#### Stage 1: Entrance Projection (Down-Projection)

* **Action:** Convert the $n$-dimensional "public" input $x_k$ into an $r$-dimensional "internal" working vector $x_r$.
* **Computation:** `x_r = MLP_down(x_k)`
* **Explanation:** This is a mandatory first step. $x_k$ ($n$-dim) cannot directly interact with the $r$-dim sandbox $H$. We must first "translate" it into the memory core's "internal language" $x_r$ ($r$-dim).

#### Stage 2: Parallel Generation of All "Instructions"

Now, **all** controller MLPs receive this **lightweight** $r$-dimensional vector $x_r$ as input.

* **Read Controller:**
    * `T_r = MLP_R(x_r)`
    * **(Explanation:** Outputs $K_r$ **read positions** `t_read`. Data flow: `r -> K_r`.)

* **Forget Controller:**
    * `P_f = MLP_F(x_r)`
    * **(Explanation:** Outputs $K_f$ **forget packets**. Each "packet" contains `t_f` (position) and `s_f` (strength). Data flow: `r -> K_f * 2`.)

* **Write & Gate Controllers:**
    * **(Explanation:** This can be one or more MLPs. Since we are now working in the efficient $r$-space, we can afford the "**independent write**" scheme, which is far simpler and more powerful than "low-rank generation".)
    * `T_w = MLP_W(x_r)`
        * **(Explanation:** Outputs $K_w$ **write positions** `t_write`. Data flow: `r -> K_w`.)
    * `V_candidates = MLP_V(x_r)`
        * **(Explanation:** Outputs $K_w$ **independent** $r$-dimensional "candidate values" `v_c`. Data flow: `r -> K_w * r`. This cost is now acceptable.)
    * `G_inputs = MLP_I(x_r)`
        * **(Explanation:** Outputs $K_w$ **independent** $r$-dimensional "write gates" `g_input`. Data flow: `r -> K_w * r`.)
    * `g_output = MLP_O(x_r)`
        * **(Explanation:** Outputs **one** $r$-dimensional "read gate" `g_output`. Data flow: `r -> r`.)

#### Stage 3: Execute "Read" Operation

Retrieve information from $H_{k-1}$ (the $M \times r$ sandbox from the previous step).

1.  Initialize an $r$-dimensional aggregate vector `v_agg` to zero.
2.  **For each read position $t_r$ in $T_r$:**
    * **Compute weights:** `i_floor = floor(t_r)`, `i_ceil = ceil(t_r)`.
    * `w_floor = 1.0 - (t_r - i_floor)`, `w_ceil = t_r - i_floor`.
    * **Interpolated Read:** `v_r = H_{k-1}[i_floor] * w_floor + H_{k-1}[i_ceil] * w_ceil`
    * **(Explanation:** This is a weighted sum of $r$-dimensional vectors, very fast.)
    * **Aggregate:** `v_agg = v_agg + v_r`.
3.  **(Explanation:** `v_agg` is now $r$-dimensional, representing all "raw clues" retrieved from memory.)

#### Stage 4: Generate "Internal Output"

* **Action:** Filter the retrieved information using the "read gate".
* **Computation:** `y_r = v_agg * g_output` (element-wise multiplication).
* **(Explanation:** `y_r` is an $r$-dimensional vector, representing the memory core's "final thought" at timestep $k$.)

#### Stage 5: Exit Projection (Up-Projection)

* **Action:** Convert the $r$-dimensional "internal result" $y_r$ back into an $n$-dimensional "public" output $y_k$.
* **Computation:** `y_k = MLP_up(y_r)`
* **(Explanation:** $y_k$ is the **final output** of this unit at timestep $k$. It is now $n$-dimensional and can be used by the next layer (if stacked) or the final prediction head (Softmax).)

#### Stage 6: Execute "Forget" Operation

* **Action:** Update the sandbox, removing old information.
* `H_temp = H_{k-1}` (Copy the $M \times r$ old sandbox).
* **For each forget packet `(t_f, s_f)` in $P_f$:**
    * `i_floor = floor(t_f)`, `i_ceil = ceil(t_f)`
    * `s_floor = s_f * (1.0 - (t_f - i_floor))`
    * `s_ceil = s_f * (t_f - i_floor)`
    * **Apply Forget (Scatter-Multiply):**
        * `H_temp[i_floor] = H_temp[i_floor] * (1.0 - s_floor)`
        * `H_temp[i_ceil] = H_temp[i_ceil] * (1.0 - s_ceil)`
    * **(Explanation:** This is an $r$-dimensional vector multiplication, very fast.)

#### Stage 7: Execute "Write" Operation

* **Action:** Add new information to $H_{\text{temp}}$ to generate $H_k$.
* `H_k = H_temp` (Start from the forgotten sandbox).
* **For the $i$-th write head (out of $K_w$):**
    1.  **Get Instructions:** Get `t_w = T_w[i]`, `v_c = V_candidates[i]`, `g_i = G_inputs[i]` from Stage 2.
    2.  **Apply Gate:** `v_final_write = v_c * g_i` (an $r$-dimensional vector).
    3.  **Compute Weights:** `i_floor = floor(t_w)`, `i_ceil = ceil(t_w)`.
    4.  `w_floor = 1.0 - (t_w - i_floor)`, `w_ceil = t_w - i_floor`.
    5.  **Apply Write (Scatter-Add):**
        * `H_k[i_floor] = H_k[i_floor] + v_final_write * w_floor`
        * `H_k[i_ceil] = H_k[i_ceil] + v_final_write * w_ceil`
    * **(Explanation:** This is an $r$-dimensional vector addition, very fast. All $K_w$ head writes are accumulated.)

#### Stage 8: Advance
* $H_k$ (the new $M \times r$ sandbox) and $y_k$ (the $n$-dimensional output) have been generated.
* The model advances to timestep $k+1$, passing $H_k$ as $H_{k-1}$ into the next loop.



# Simulated Smooth RNN

Simulated Smooth RNN (SS-RNN)，这是一种用于序列处理的循环架构。其核心创新是一种内存机制，它以恒定的 O(1) 计算成本实现了灵活的、连续空间的寻址能力。SS-RNN 使用一个控制器为其多头读/写系统生成浮点数地址（例如 50.6）。这些“平滑”(Smooth) 的地址继而通过一个可微分的线性插值操作进行“模拟”(Simulated)，该操作仅与两个最近的离散内存槽位交互，即可执行所有内存动作。



### 一、 高层概念

这是一个**循环（Recurrent）**记忆单元，用于存取扫描序列时产生的状态。

1.  **外部维度 ($n$)：** 模型的“公共接口”维度（例如 $n=768$）。它接收 $n$ 维的 $x_k$（输入）并产生 $n$ 维的 $y_k$（输出）。
2.  **内部维度 ($r$)：** 模型的“核心记忆”维度（例如 $r=64$）。$r$ 远小于 $n$ ($r \ll n$)。

所有复杂、昂贵的内存操作（读、写、遗忘）都发生在这个**轻量级的 $r$ 维空间**中，以此实现极高的计算和内存效率。模型在“入口”处将 $n$ 维输入**降维**到 $r$ 维，在“出口”处将 $r$ 维结果**升维**回 $n$ 维。

### 二、 核心组件与符号定义

* **$k$**: 当前时间步。
* **$x_k$**: $n$ 维的输入嵌入向量（例如 $n=768$）。
* **$y_k$**: $n$ 维的输出向量。
* **$H_k$**: **内存沙盘**。这是一个 $M \times r$ 的张量（$M$ 个槽位，每个槽位 $r$ 维）。
* **$M$**: 内存沙盘的槽位总数（超参数，例如 $M=1000$）。
* **$n$**: 外部维度（例如 $n=768$）。
* **$r$**: **内部核心维度**（超参数，例如 $r=64$）。
* **$\text{MLP}_{\text{down}}$**: 入口投影 MLP，负责 $n \to r$ 的降维。
* **$\text{MLP}_{\text{up}}$**: 出口投影 MLP，负责 $r \to n$ 的升维。
* **$\text{MLP}_{\text{R/F/W/G}}$**: 各种“控制器”MLP，它们现在全部在 $r$ 维空间中运行。
* **$K_r, K_w, K_f$**: 读取、写入、遗忘操作“头” (Head) 的数量。
* **$t$**: 一个操作位置（标量），范围 `[0, M-1]`。
* **$v$**: 一个 $r$ 维的值向量。
* **$g$**: 一个 $r$ 维的门控向量（值 0-1）。


### 三、 详细的单步（k 时刻）前向传播

在 $k$ 时刻，模型接收 $n$ 维的 $x_k$ 和 $M \times r$ 的沙盘状态 $H_{k-1}$。

#### 阶段 1：入口投影 (Down-Projection)

* **动作：** 将 $n$ 维的“公共”输入 $x_k$ 转换为 $r$ 维的“内部”工作向量 $x_r$。
* **计算：** `x_r = MLP_down(x_k)`
* **说明：** 这是必须的第一步。$x_k$（$n$ 维）无法直接与 $r$ 维的沙盘 $H$ 交互。我们必须先将其“翻译”成内存核心的“内部语言” $x_r$（$r$ 维）。

#### 阶段 2：并行生成所有“指令”

现在，**所有**的控制器 MLP 都接收这个**轻量级**的 $r$ 维向量 $x_r$ 作为输入。

* **读取控制器 (Read Controller)：**
    * `T_r = MLP_R(x_r)`
    * **(说明：** 输出 $K_r$ 个**读取位置** `t_read`。数据流：`r -> K_r`。)

* **遗忘控制器 (Forget Controller)：**
    * `P_f = MLP_F(x_r)`
    * **（说明：** 输出 $K_f$ 个**遗忘包**。每个“包”包含 `t_f` (位置) 和 `s_f` (强度)。数据流：`r -> K_f * 2`。)

* **写入/门控控制器 (Write & Gate Controllers)：**
    * **（说明：** 这可以是一个或多个 MLP。由于我们现在工作在高效的 $r$ 维空间，我们可以负担得起“**独立写入**”方案，这远比“低秩生成”更简单、更强大。)
    * `T_w = MLP_W(x_r)`
        * **(说明：** 输出 $K_w$ 个**写入位置** `t_write`。数据流：`r -> K_w`。)
    * `V_candidates = MLP_V(x_r)`
        * **（说明：** 输出 $K_w$ 个**独立**的 $r$ 维“候选值” `v_c`。数据流：`r -> K_w * r`。这个成本现在是可接受的。)
    * `G_inputs = MLP_I(x_r)`
        * **（说明：** 输出 $K_w$ 个**独立**的 $r$ 维“写入门” `g_input`。数据流：`r -> K_w * r`。)
    * `g_output = MLP_O(x_r)`
        * **（说明：** 输出**一个** $r$ 维的“读取门” `g_output`。数据流：`r -> r`。)

#### 阶段 3：执行“读取”操作

从 $H_{k-1}$ （上一步的 $M \times r$ 沙盘）中检索信息。

1.  初始化一个 $r$ 维的聚合向量 `v_agg` 为零。
2.  **对于 $T_r$ 中的每一个读取位置 $t_r$：**
    * **计算权重：** `i_floor = floor(t_r)`, `i_ceil = ceil(t_r)`。
    * `w_floor = 1.0 - (t_r - i_floor)`, `w_ceil = t_r - i_floor`。
    * **插值读取：** `v_r = H_{k-1}[i_floor] * w_floor + H_{k-1}[i_ceil] * w_ceil`
    * **（说明：** 这是一个 $r$ 维向量的加权求和，非常快速。)
    * **聚合：** `v_agg = v_agg + v_r`。
3.  **（说明：** `v_agg` 现在是 $r$ 维的，代表了从内存中检索到的所有“原始线索”。)

#### 阶段 4：生成“内部输出”

* **动作：** 使用“读取门”过滤检索到的信息。
* **计算：** `y_r = v_agg * g_output` (逐元素乘法)。
* **（说明：** `y_r` 是一个 $r$ 维向量，代表内存核心在 $k$ 时刻的“最终思考结果”。)

#### 阶段 5：出口投影 (Up-Projection)

* **动作：** 将 $r$ 维的“内部结果” $y_r$ 转换回 $n$ 维的“公共”输出 $y_k$。
* **计算：** `y_k = MLP_up(y_r)`
* **（说明：** $y_k$ 是这个单元在 $k$ 时刻的**最终输出**，它现在是 $n$ 维的，可以被下一层（如果是堆叠的）或最终的预测头（Softmax）所使用。)

#### 阶段 6：执行“遗忘”操作

* **动作：** 更新沙盘，移除旧信息。
* `H_temp = H_{k-1}` (复制一份 $M \times r$ 的旧沙盘)。
* **对于 $P_f$ 中的每一个遗忘包 `(t_f, s_f)`：**
    * `i_floor = floor(t_f)`, `i_ceil = ceil(t_f)`
    * `s_floor = s_f * (1.0 - (t_f - i_floor))`
    * `s_ceil = s_f * (t_f - i_floor)`
    * **应用遗忘 (Scatter-Multiply)：**
        * `H_temp[i_floor] = H_temp[i_floor] * (1.0 - s_floor)`
        * `H_temp[i_ceil] = H_temp[i_ceil] * (1.0 - s_ceil)`
    * **（说明：** 这是一个 $r$ 维向量的乘法，非常快速。)

#### 阶段 7：执行“写入”操作

* **动作：** 在 $H_{\text{temp}}$ 的基础上，添加新信息，生成 $H_k$。
* `H_k = H_temp` (从被遗忘过的沙盘开始)。
* **对于 $K_w$ 个写入头中的第 $i$ 个头：**
    1.  **获取指令：** 从阶段 2 中拿到 `t_w = T_w[i]`, `v_c = V_candidates[i]`, `g_i = G_inputs[i]`。
    2.  **应用门控：** `v_final_write = v_c * g_i` (一个 $r$ 维向量)。
    3.  **计算权重：** `i_floor = floor(t_w)`, `i_ceil = ceil(t_w)`。
    4.  `w_floor = 1.0 - (t_w - i_floor)`, `w_ceil = t_w - i_floor`。
    5.  **应用写入 (Scatter-Add)：**
        * `H_k[i_floor] = H_k[i_floor] + v_final_write * w_floor`
        * `H_k[i_ceil] = H_k[i_ceil] + v_final_write * w_ceil`
    * **（说明：** 这是一个 $r$ 维向量的加法，非常快速。所有 $K_w$ 个头的写入会累积。)

#### 阶段 8：推进
* $H_k$ (新的 $M \times r$ 沙盘) 和 $y_k$ ( $n$ 维输出) 被生成。
* 模型推进到 $k+1$ 时间步，将 $H_k$ 作为 $H_{k-1}$ 传入下一个循环。
