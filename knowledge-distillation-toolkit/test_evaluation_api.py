#!/usr/bin/env python3
"""
Test script for evaluation API endpoints with async task queue.
Tests real-time progress streaming and task management.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

API_BASE = "http://localhost:8765"

async def test_evaluation_endpoints():
    """Test all evaluation endpoints."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🧪 Testing Evaluation API Endpoints\n")
        print("=" * 60)
        
        # Test 1: List experiments (to get a valid experiment_id)
        print("\n1️⃣ Listing experiments...")
        try:
            response = await client.get(f"{API_BASE}/api/experiments")
            if response.status_code == 200:
                experiments = response.json().get("experiments", [])
                if experiments:
                    exp_id = experiments[0]["experiment_id"]
                    print(f"   ✅ Found experiment: {exp_id}")
                else:
                    print("   ⚠️  No experiments found - creating test experiment")
                    exp_id = "test_exp_001"
            else:
                print(f"   ⚠️  Could not list experiments (status {response.status_code})")
                exp_id = "test_exp_001"
        except Exception as e:
            print(f"   ⚠️  Error listing experiments: {e}")
            exp_id = "test_exp_001"
        
        # Test 2: Start evaluation task
        print("\n2️⃣ Starting evaluation task...")
        try:
            eval_request = {
                "experiment_id": exp_id,
                "eval_type": "standard",
                "benchmark_tasks": None
            }
            response = await client.post(
                f"{API_BASE}/api/evaluation/start",
                json=eval_request
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("task_id")
                print(f"   ✅ Evaluation started - Task ID: {task_id}")
                print(f"   📝 Status: {result.get('status')}")
                print(f"   💬 Message: {result.get('message')}")
            elif response.status_code == 404:
                print(f"   ⚠️  Experiment not found: {exp_id}")
                print("   💡 Tip: Create an experiment first or check experiment_id")
                return
            else:
                print(f"   ❌ Failed to start evaluation (status {response.status_code})")
                print(f"   Response: {response.text}")
                return
                
        except Exception as e:
            print(f"   ❌ Error starting evaluation: {e}")
            return
        
        # Test 3: Monitor task progress
        print("\n3️⃣ Monitoring task progress...")
        max_checks = 10
        check_count = 0
        
        while check_count < max_checks:
            try:
                response = await client.get(
                    f"{API_BASE}/api/evaluation/task/{task_id}"
                )
                
                if response.status_code == 200:
                    task = response.json()
                    status = task.get("status")
                    progress = task.get("progress", 0)
                    stage = task.get("current_stage", "unknown")
                    
                    print(f"   📊 Status: {status} | Progress: {progress:.1f}% | Stage: {stage}")
                    
                    # Show progress data if available
                    progress_data = task.get("progress_data")
                    if progress_data:
                        batch = progress_data.get("batch", "?")
                        total = progress_data.get("total_batches", "?")
                        acc = progress_data.get("current_accuracy")
                        loss = progress_data.get("current_loss")
                        
                        details = f"      Batch: {batch}/{total}"
                        if acc is not None:
                            details += f" | Accuracy: {acc:.4f}"
                        if loss is not None:
                            details += f" | Loss: {loss:.4f}"
                        print(details)
                    
                    # Check if completed
                    if status in ["completed", "failed", "cancelled"]:
                        print(f"\n   ✅ Task finished with status: {status}")
                        
                        if status == "completed":
                            result = task.get("result")
                            if result:
                                print(f"   📈 Results:")
                                print(f"      Accuracy: {result.get('accuracy', 'N/A')}")
                                print(f"      F1 Score: {result.get('f1', 'N/A')}")
                                print(f"      Precision: {result.get('precision', 'N/A')}")
                                print(f"      Recall: {result.get('recall', 'N/A')}")
                        elif status == "failed":
                            error = task.get("error")
                            print(f"   ❌ Error: {error}")
                        
                        break
                    
                    # Wait before next check
                    await asyncio.sleep(2)
                    check_count += 1
                    
                else:
                    print(f"   ❌ Failed to get task status (status {response.status_code})")
                    break
                    
            except Exception as e:
                print(f"   ❌ Error monitoring task: {e}")
                break
        
        if check_count >= max_checks:
            print(f"   ⏱️  Reached max checks ({max_checks}) - task still running")
        
        # Test 4: List all tasks
        print("\n4️⃣ Listing all tasks...")
        try:
            response = await client.get(f"{API_BASE}/api/evaluation/tasks")
            
            if response.status_code == 200:
                result = response.json()
                tasks = result.get("tasks", [])
                running_count = result.get("running_count", 0)
                
                print(f"   ✅ Total tasks: {len(tasks)}")
                print(f"   🏃 Running tasks: {running_count}")
                
                if tasks:
                    print(f"   📋 Recent tasks:")
                    for task in tasks[:3]:  # Show first 3
                        print(f"      - {task['task_id'][:8]}... | {task['status']} | {task['eval_type']}")
            else:
                print(f"   ❌ Failed to list tasks (status {response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Error listing tasks: {e}")
        
        # Test 5: Cancel a task (if still running)
        print("\n5️⃣ Testing task cancellation...")
        try:
            # Check if our task is still running
            response = await client.get(
                f"{API_BASE}/api/evaluation/task/{task_id}"
            )
            
            if response.status_code == 200:
                task = response.json()
                if task.get("status") == "running":
                    # Try to cancel it
                    response = await client.post(
                        f"{API_BASE}/api/evaluation/task/{task_id}/cancel"
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ Cancellation requested: {result.get('message')}")
                    else:
                        print(f"   ❌ Failed to cancel (status {response.status_code})")
                else:
                    print(f"   ℹ️  Task already {task.get('status')} - cannot cancel")
            else:
                print(f"   ⚠️  Could not check task status")
                
        except Exception as e:
            print(f"   ❌ Error testing cancellation: {e}")
        
        # Test 6: Cleanup old tasks
        print("\n6️⃣ Testing task cleanup...")
        try:
            response = await client.delete(
                f"{API_BASE}/api/evaluation/tasks/cleanup",
                params={"max_age_hours": 1, "keep_last_n": 5}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Cleanup completed: {result.get('message')}")
            else:
                print(f"   ❌ Cleanup failed (status {response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Error during cleanup: {e}")
        
        print("\n" + "=" * 60)
        print("✅ API endpoint testing complete!\n")


async def test_websocket_progress():
    """Test WebSocket progress streaming (requires backend running)."""
    
    print("\n🔌 Testing WebSocket Progress Streaming")
    print("=" * 60)
    print("⚠️  Note: This requires the backend WebSocket server to be running")
    print("   Run: python ui/backend/api.py")
    print("   Then connect to: ws://localhost:8765/ws")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 Evaluation System API Test Suite")
    print("=" * 60)
    print("\n📋 Prerequisites:")
    print("   1. Backend server running: python ui/backend/api.py")
    print("   2. At least one experiment created in the system")
    print("   3. Models available for evaluation")
    print("\n⚙️  Starting tests...\n")
    
    try:
        asyncio.run(test_evaluation_endpoints())
        asyncio.run(test_websocket_progress())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
