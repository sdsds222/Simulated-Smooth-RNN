
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
    * `T_r = MLP_R(x_r，H_{k-1})`
    * **(Explanation:** Outputs $K_r$ **read positions** `t_read`. Data flow: `r -> K_r`.)

* **Forget Controller:**
    * `P_f = MLP_F(x_r，H_{k-1})`
    * **(Explanation:** Outputs $K_f$ **forget packets**. Each "packet" contains `t_f` (position) and `s_f` (strength). Data flow: `r -> K_f * 2`.)

* **Write & Gate Controllers:**
    * **(Explanation:** This can be one or more MLPs. Since we are now working in the efficient $r$-space, we can afford the "**independent write**" scheme, which is far simpler and more powerful than "low-rank generation".)
    * `T_w = MLP_W(x_r，H_{k-1})`
        * **(Explanation:** Outputs $K_w$ **write positions** `t_write`. Data flow: `r -> K_w`.)
    * `V_candidates = MLP_V(x_r，H_{k-1})`
        * **(Explanation:** Outputs $K_w$ **independent** $r$-dimensional "candidate values" `v_c`. Data flow: `r -> K_w * r`. This cost is now acceptable.)
    * `G_inputs = MLP_I(x_r，H_{k-1})`
        * **(Explanation:** Outputs $K_w$ **independent** $r$-dimensional "write gates" `g_input`. Data flow: `r -> K_w * r`.)
    * `g_output = MLP_O(x_r，H_{k-1})`
        * **(Explanation:** Outputs **one** $r$-dimensional "read gate" `g_output`. Data flow: `r -> r`.)

#### Stage 3: Execute "Read" Operation

Retrieve information from $H_{k-1}$ (the $M \times r$ sandbox from the previous step).

**For each read position $t_r$ in $T_r$:**
    * **Compute weights:** `i_floor = floor(t_r)`, `i_ceil = ceil(t_r)`.
    * `w_floor = 1.0 - (t_r - i_floor)`, `w_ceil = t_r - i_floor`.
    * **Interpolated Read:** `v_r = H_{k-1}[i_floor] * w_floor + H_{k-1}[i_ceil] * w_ceil`
    * **(Explanation:** This is a weighted sum of $r$-dimensional vectors, very fast.)
* For each (t_r \in T_r): compute two-point interpolated read (v^{(j)}\in \mathbb{R}^r) from (H_{k-1}) using floor/ceil and linear weights.
* Stack reads: `V_stack = [v^(1), ..., v^(K_r)]` with shape `[K_r, r]`.
* Concatenate: `v_cat = reshape(V_stack, K_r * r)`.

#### Stage 4: Generate "Internal Output"

* **Action:** Filter the retrieved information using the "read gate".
* Read gate: `g_read = MLP_O(z_k)` with shape `K_r * r`, range `[0, 1]`.
* Elementwise gating: `y_cat = v_cat * g_read`.

#### Stage 5: Exit Projection (Up-Projection)

* **Action:** Convert "internal result" $y_cat$ back into an $n$-dimensional "public" output $y_k$.
* After gating, take y_cat with shape K_r*r, feed it to MLP_up, and map to n to obtain y_k.

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

## Alternatives

### Architecture A: RNN Instruction

1. Controller update
   Use an RNN to take the previous hidden state and the current input to produce h_k

2. Downward mapping
   Use an MLP to map the external input x_k into the internal vector x_r

3. Assemble control context
   Concatenate x_r and h_k to obtain z_k as the control-signal input

4. Generate read instructions
   Use MLP_R to output the list of continuous read coordinates T_r from z_k
   Use MLP_O to output the read gate g_out from z_k

5. Execute read
   For each coordinate t_r, perform linear interpolation on the two adjacent slots of the sandbox H to obtain the read value
   Aggregate the read values and use g_out for elementwise gating to obtain y_r
   Use MLP_up to up-project y_r to obtain y_k

6. Generate forget instructions
   Use MLP_F_pos to output the list of continuous forget coordinates T_f from z_k
   Use MLP_F_str to output the forget strength S_f from z_k

7. Execute forget
   For each coordinate t_f, apply elementwise multiplicative decay on the two adjacent slots to obtain the temporary sandbox H_tmp

8. Execute write
   Keep the original write path
   Perform two-point interpolated additive writes on H_tmp to obtain H_k

9. Advance the timeline
   Output y_k
   Update the states to H_k and h_k

### Architecture B: Parallel

This solution is a **Batch** architecture that is mathematically (during training) **parallelizable**.

**1. Core Formula (Controller):**
The controller's decision **depends only on the current input**.
* `Controls_k = MLP_Control(x_k)`

**2. Mechanism (Parallel):**
This solution requires an "**All-Addition**" logic (i.e., the "forget" operation is implemented as "adding a negative vector").
1.  **Parallel Delta Generation:** All $N$ "controller" `MLP(x_k)` modules run in parallel. Looking only at $x_k$, they generate a "sandbox delta" `H_delta_k` (an $M \times r$ set of add/subtract instructions) for each of the $N$ timesteps.
2.  **Parallel Scan:** The model uses a "Prefix Sum" algorithm to compute $N$ "causal history snapshots" in parallel.
    * `H_read_k = H_delta_1 + H_delta_2 + ... + H_delta_k`
3.  **Parallel Reads:** All $N$ "Read" operations `Read(H_read_k, Controls_k)` run in parallel, generating all $N$ outputs $y_k$.

**3. Advantages (Pros):**
* **Parallelizable Training:** The architecture (due to its "stateless" controller and "all-addition" operation) is mathematically compatible with parallel scan. This allows it to compete with Mamba in training speed.
* **Conceptually Simple:** Removes the need for the "second" memory system (the RNN navigator).

**4. Disadvantages (Cons):**
* **"Blind" Controller:** This is the most critical weakness. At $k=50$, `MLP(x_50)` does not know what happened in the sandbox at $k=10$.
* **Cannot Dynamically Solve "Write Conflicts":** The controller cannot "check" if an address is already occupied. If `MLP("apple")` and `MLP("orange")` both (blindly) learn to hash to `t=50.6`, they will inevitably be "blended" together (`V_APPLE + V_ORANGE`).
* **Extremely High Training Difficulty:** The controller must, in a "blind" state, learn a "globally optimal hashing/clustering scheme" based only on $x_k$ and the final loss signal.
* **"Subtractive Forget" Paradox:** The "blind" "Forget Controller" `MLP_F(x_k)` logically cannot know *which* vector it is supposed to "subtract" (e.g., `V_APPLE`), because it has never "read" the sandbox.


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
* **计算：** `x_r = MLP_down(x_k，H_{k-1})`
* **说明：** 这是必须的第一步。$x_k$（$n$ 维）无法直接与 $r$ 维的沙盘 $H$ 交互。我们必须先将其“翻译”成内存核心的“内部语言” $x_r$（$r$ 维）。

#### 阶段 2：并行生成所有“指令”

现在，**所有**的控制器 MLP 都接收这个**轻量级**的 $r$ 维向量 $x_r$ 作为输入。

* **读取控制器 (Read Controller)：**
    * `T_r = MLP_R(x_r，H_{k-1})`
    * **(说明：** 输出 $K_r$ 个**读取位置** `t_read`。数据流：`r -> K_r`。)

* **遗忘控制器 (Forget Controller)：**
    * `P_f = MLP_F(x_r，H_{k-1})`
    * **（说明：** 输出 $K_f$ 个**遗忘包**。每个“包”包含 `t_f` (位置) 和 `s_f` (强度)。数据流：`r -> K_f * 2`。)

* **写入/门控控制器 (Write & Gate Controllers)：**
    * **（说明：** 这可以是一个或多个 MLP。由于我们现在工作在高效的 $r$ 维空间，我们可以负担得起“**独立写入**”方案，这远比“低秩生成”更简单、更强大。)
    * `T_w = MLP_W(x_r，H_{k-1})`
        * **(说明：** 输出 $K_w$ 个**写入位置** `t_write`。数据流：`r -> K_w`。)
    * `V_candidates = MLP_V(x_r，H_{k-1})`
        * **（说明：** 输出 $K_w$ 个**独立**的 $r$ 维“候选值” `v_c`。数据流：`r -> K_w * r`。这个成本现在是可接受的。)
    * `G_inputs = MLP_I(x_r，H_{k-1})`
        * **（说明：** 输出 $K_w$ 个**独立**的 $r$ 维“写入门” `g_input`。数据流：`r -> K_w * r`。)
    * `g_output = MLP_O(x_r，H_{k-1})`
        * **（说明：** 输出**一个** $r$ 维的“读取门” `g_output`。数据流：`r -> r`。)

#### 阶段 3：执行“读取”操作

从 $H_{k-1}$ （上一步的 $M \times r$ 沙盘）中检索信息。

**对于 $T_r$ 中的每一个读取位置 $t_r$：**
    * **计算权重：** `i_floor = floor(t_r)`, `i_ceil = ceil(t_r)`。
    * `w_floor = 1.0 - (t_r - i_floor)`, `w_ceil = t_r - i_floor`。
    * **插值读取：** `v_r = H_{k-1}[i_floor] * w_floor + H_{k-1}[i_ceil] * w_ceil`
    * **（说明：** 这是一个 $r$ 维向量的加权求和，非常快速。)
* 对每个读取坐标 (t_r) 用两点插值从 (H_{k-1}) 取出 (r) 维向量 (v^{(j)})。
* 堆叠为矩阵：`V_stack = [v^(1), ..., v^(K_r)]`，形状 `[K_r, r]`。
* 展平拼接：`v_cat = reshape(V_stack, K_r * r)`。

#### 阶段 4：生成“内部输出”

* **动作：** 使用“读取门”过滤检索到的信息。
* 读取门：`g_read = MLP_O(z_k)`，形状 `K_r * r`，范围 `[0,1]`。
* 逐元素门控：`y_cat = v_cat * g_read`。

#### 阶段 5：出口投影 (Up-Projection)

* **动作：** 将 “内部结果” $y_cat$ 转换回 $n$ 维的“公共”输出 $y_k$。
* 门控后向量 y_cat 维度 K_r*r
* 送入 MLP_up 映射到维度 n 得到 y_k

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

## 可选

### 架构A：RNN指导
 
1. 控制器更新
   用 RNN 接收上一步隐藏态与本步输入生成 h_k

2. 降维映射
   用 MLP 把外部输入 x_k 映射为内部向量 x_r

3. 组装控制上下文
   将 x_r 与 h_k 拼接得到 z_k 作为控制信号输入

4. 生成读取指令
   用 MLP_R 由 z_k 输出连续读取坐标列表 T_r
   用 MLP_O 由 z_k 输出读取门 g_out

5. 执行读取
   对每个坐标 t_r 在沙盘 H 的相邻两格做线性插值得到读值
   聚合读值并用 g_out 做逐元素门控得到 y_r
   用 MLP_up 将 y_r 升维得到 y_k

6. 生成遗忘指令
   用 MLP_F_pos 由 z_k 输出连续遗忘坐标列表 T_f
   用 MLP_F_str 由 z_k 输出遗忘强度 S_f

7. 执行遗忘
   对每个坐标 t_f 在相邻两格做逐元素乘性衰减得到临时沙盘 H_tmp

8. 执行写入
   维持原有写入路径
   在 H_tmp 上做两点插值的加法写入得到 H_k

9. 推进时序
   输出 y_k
   状态更新为 H_k 与 h_k


### 架构B：并行

此方案是一个批处理 (Batch) 架构，它在数学上可以（在训练时）并行化。

**1. 核心公式 (控制器)：**
控制器的决策**仅依赖于当前输入**。
* `Controls_k = MLP_Control(x_k)`

**2. 机制 (并行)：**
此方案需要一个“**全加法**”逻辑（即“遗忘”操作被实现为“加一个负向量”）。
1.  **并行生成增量 (Deltas)：** 所有 $N$ 个“控制器”`MLP(x_k)` 并行运行。只看 $x_k$，为 $N$ 个时间步中的每一步都生成一个“沙盘增量” `H_delta_k`（一个 $M \times r$ 的加法/减法指令集）。
2.  **并行扫描 (Scan)：** 模型使用一个“前缀和”（Prefix Sum）算法，并行计算出 $N$ 个“因果历史快照”。
    * `H_read_k = H_delta_1 + H_delta_2 + ... + H_delta_k`
3.  **并行读取 (Reads)：** 所有 $N$ 个“读取”操作 `Read(H_read_k, Controls_k)` 并行运行，生成所有 $N$ 个输出 $y_k$。

**3. 优点 (Pros)：**
* **可并行训练：** 架构（由于其“无状态依赖”的控制器和“全加法”操作）在数学上与“并行扫描”兼容。这使其在训练速度上可以与 Mamba 竞争。
* **概念上简单：** 移除了对“第二个”记忆系统（RNN 导航仪）的需求。

**4. 缺点 (Cons)：**
* **“盲目”的控制器 (Blind Controller)：** 这是最致命的弱点。在 $k=50$ 时，`MLP(x_50)` 不知道 $k=10$ 时在沙盘上发生了什么。
* **无法动态解决“写入冲突”：** 控制器无法“检查”一个地址是否已被占用。如果 `MLP("apple")` 和 `MLP("orange")` 都（盲目地）学会了哈希到 `t=50.6`，它们将不可避免地被“搅拌”在一起（`V_APPLE + V_ORANGE`）。
* **训练难度极高：** 控制器必须在“盲目”的情况下，仅凭 $x_k$ 和最终的“损失信号”，就学会一个“全局最优的哈希/聚类方案”。
* **“减法遗忘”的悖论：** “盲目”的“遗忘控制器” `MLP_F(x_k)` 在逻辑上无法知道它应该去“减”哪个向量（例如 `V_APPLE`），因为它从未“读取”过沙盘。
