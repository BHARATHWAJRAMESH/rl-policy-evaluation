# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by the policy at state s
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:
### POLICY 1:
<img width="563" height="162" alt="Screenshot 2025-09-27 111651" src="https://github.com/user-attachments/assets/c7b5117a-e4da-41f6-984c-11b0c85ab422" />
<img width="552" height="134" alt="Screenshot 2025-09-27 111703" src="https://github.com/user-attachments/assets/649ba86c-5d4c-4d33-bdd7-335077946239" />

### POLICY 2:
<img width="563" height="162" alt="Screenshot 2025-09-27 111651" src="https://github.com/user-attachments/assets/c7b5117a-e4da-41f6-984c-11b0c85ab422" />
<img width="709" height="110" alt="Screenshot 2025-09-27 111924" src="https://github.com/user-attachments/assets/0daeaa29-a5bf-4e34-8a70-9d673f77b2ac" />

### COMPARISON:
<img width="508" height="195" alt="Screenshot 2025-09-27 111749" src="https://github.com/user-attachments/assets/6bc97a71-6403-4437-96fc-6140789700cc" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
