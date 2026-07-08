import multiprocessing
import asyncio
import time
import random
import cv2  # 需要安裝 opencv-python
from queue import Empty

# ==========================================
# 1. 模擬/實際的工具函式 (請根據需求替換)
# ==========================================

def detect_slide_change(prev_frame, curr_frame):
    """
    使用 OpenCV 偵測換頁 (實際開發建議使用 SSIM 或直方圖比較)
    """
    if prev_frame is None:
        return True
    # 簡單的差異計算
    diff = cv2.absdiff(prev_frame, curr_frame)
    return diff.mean() > 30  # 閾值需根據影片調整

def heavy_image_processing(frame_id, frame_data):
    """
    [計算密集型] 模擬超解析度 (SR) 或 影像增強
    在實際開發中，這裡會呼叫 Real-ESRGAN 或 OpenCV 濾波器
    """
    print(f"[Worker-{multiprocessing.current_process().name}] 正在對第 {frame_id} 幀進行超解析度重建...")
    # 模擬耗時的運算 (例如 GPU 運算)
    time.sleep(random.uniform(1.0, 2.0)) 
    
    # 模擬處理後的結果 (實際應儲存檔案或回傳 numpy array)
    return f"processed_slide_{frame_id}.jpg"

async def call_gemini_api(image_path):
    """
    [I/O 密集型] 模擬呼叫 Gemini API
    """
    print(f"  [Gemini-Async] 正在發送 API 請求: {image_path}")
    # 模擬網路延遲 (API 回傳時間)
    await asyncio.sleep(random.uniform(1.5, 3.0))
    return f"【摘要】這是來自 {image_path} 的結構化數據內容。"

# ==========================================
# 2. 核心工作進程 (Workers)
# ==========================================

def image_worker(input_queue, output_queue):
    """
    多進程 Worker：從 input_queue 拿取原始幀，做影像處理，放進 output_queue
    """
    while True:
        try:
            # 等待任務，timeout 避免死鎖
            task = input_queue.get(timeout=5)
            if task == "STOP":
                break
            
            frame_id, frame_data = task
            # 執行重型運算
            processed_img_path = heavy_image_processing(frame_id, frame_data)
            
            # 將結果丟回輸出隊列
            output_queue.put((frame_id, processed_img_path))
        except Empty:
            continue
        except Exception as e:
            print(f"Worker Error: {e}")

# ==========================================
# 3. 異步管理器 (Async Manager)
# ==========================================

async def gemini_manager(output_queue):
    """
    異步管理器：負責監控 output_queue 並非同步發送 API 請求
    """
    print("[Manager] Gemini 異步管理器已啟動...")
    active_tasks = set()
    
    while True:
        # 1. 檢查是否有新處理好的圖片進入 Queue
        try:
            # 使用 loop.run_in_executor 避免在 async 裡使用阻塞的 queue.get()
            # 或者簡單使用 get_nowait()
            if not output_queue.empty():
                frame_id, img_path = output_queue.get_nowait()
                
                # 啟動非同步 API 任務
                task = asyncio.create_task(call_gemini_api(img_path))
                active_tasks.add(task)
                
                # 設定任務完成後的 Callback
                def task_done_callback(t):
                    try:
                        result = t.result()
                        print(f"\n>>> [SUCCESS] {result}\n")
                    except Exception as e:
                        print(f"API Task Failed: {e}")
                    finally:
                        active_tasks.discard(t)

                task.add_done_callback(task_done_callback)
            
        except Empty:
            pass

        # 2. 如果沒有新任務，且所有 API 任務都結束了，則結束管理器
        if not active_tasks and output_queue.empty():
            # 這裡需要一個機制來判斷「影片是否真的讀完了」
            # 為了範例簡潔，我們檢查一個外部狀態，實際開發建議用 Event
            pass

        await asyncio.sleep(0.1) # 釋放控制權給其他 task

# ==========================================
# 4. 主協調器 (Main Orchestrator)
# ==========================================

async def main_orchestrator(video_path):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    
    # A. 啟動多進程池 (例如使用 4 個核心做影像處理)
    num_workers = 4
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=image_worker, args=(input_queue, output_queue), name=f"Proc-{i}")
        p.start()
        processes.append(p)

    # B. 啟動異步 Gemini 管理器
    manager_task = asyncio.create_task(gemini_manager(output_queue))

    # C. 模擬影片讀取與偵測 (Producer)
    print(f"[Main] 開始讀取影片: {video_path}")
    cap = cv2.VideoCapture(video_path) # 實際開發請打開影片
    
    # 為了演示，我們改用模擬數據
    prev_frame = None
    for frame_id in range(1, 21):
        # 模擬讀取一幀 (實際應使用 cap.read())
        curr_frame = None # 這裡應是 cv2.imread 或 cap.read()
        
        # 模擬換頁偵測
        if frame_id % 4 == 0: # 每 4 幀觸發一次換頁
            print(f"[Main] 偵測到換頁事件！Frame: {frame_id}")
            # 模擬傳入一幀的數據 (實際應傳入 numpy array)
            dummy_frame = None 
            input_queue.put((frame_id, dummy_frame))
            
        await asyncio.sleep(0.2) # 模擬影片播放速度

    # D. 結束流程
    print("[Main] 影片讀取完畢，正在清理資源...")
    
    # 1. 停止 Worker 進程
    for _ in range(num_workers):
        input_queue.put("STOP")
    for p in processes:
        p.join()
    
    # 2. 等待 API 任務完成 (這裡簡單等待一段時間，實際應使用 asyncio.Event)
    print("[Main] 等待剩餘 API 請求回傳...")
    await asyncio.sleep(10) 
    
    manager_task.cancel()
    print("[Main] 系統已安全關閉。")

if __name__ == "__main__":
    # Windows 環境必須在 if __name__ == "__main__": 下執行
    asyncio.run(main_orchestrator("dummy_video.mp4"))
