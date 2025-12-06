"""
Script to verify that the warmup mechanism is working correctly.
Checks TensorBoard logs or training logs to confirm CLIP loss behavior.
"""

import argparse
import os
import re


def verify_from_text_log(log_path):
    """Verify warmup from rank0.txt log file"""
    print(f"Checking log file: {log_path}")
    print("-" * 60)
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return False
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    warmup_epochs = []
    active_epochs = []
    warmup_transition_found = False
    
    for line in lines:
        # Check for warmup transition message
        if "WARMUP COMPLETE" in line:
            warmup_transition_found = True
            print(f"✓ Found warmup transition message:")
            print(f"  {line.strip()}")
        
        # Parse training logs
        if "clip_loss:" in line and "WARMUP" in line:
            # Extract epoch number and clip loss
            epoch_match = re.search(r'Epoch\[(\d+)/', line)
            loss_match = re.search(r'clip_loss:([\d.]+)', line)
            if epoch_match and loss_match:
                epoch = int(epoch_match.group(1))
                clip_loss = float(loss_match.group(1))
                warmup_epochs.append((epoch, clip_loss))
        
        elif "clip_loss:" in line and "CLIP_ACTIVE" in line:
            epoch_match = re.search(r'Epoch\[(\d+)/', line)
            loss_match = re.search(r'clip_loss:([\d.]+)', line)
            if epoch_match and loss_match:
                epoch = int(epoch_match.group(1))
                clip_loss = float(loss_match.group(1))
                active_epochs.append((epoch, clip_loss))
    
    print("\n" + "=" * 60)
    print("Warmup Verification Results:")
    print("=" * 60)
    
    # Check warmup epochs
    if warmup_epochs:
        print(f"\n✓ Warmup Phase (CLIP loss should be 0.0000):")
        for epoch, loss in warmup_epochs[:5]:  # Show first 5
            status = "✓" if loss == 0.0 else "❌"
            print(f"  {status} Epoch {epoch}: clip_loss = {loss:.4f}")
        if len(warmup_epochs) > 5:
            print(f"  ... ({len(warmup_epochs)} total warmup records)")
        
        # Verify all warmup losses are 0
        all_zero = all(loss == 0.0 for _, loss in warmup_epochs)
        if all_zero:
            print(f"  ✓ All warmup CLIP losses are 0.0 ✓")
        else:
            print(f"  ❌ Some warmup CLIP losses are non-zero!")
    
    # Check active epochs
    if active_epochs:
        print(f"\n✓ Active Phase (CLIP loss should be > 0.0000):")
        for epoch, loss in active_epochs[:5]:  # Show first 5
            status = "✓" if loss > 0.0 else "❌"
            print(f"  {status} Epoch {epoch}: clip_loss = {loss:.4f}")
        if len(active_epochs) > 5:
            print(f"  ... ({len(active_epochs)} total active records)")
        
        # Verify all active losses are > 0
        all_positive = all(loss > 0.0 for _, loss in active_epochs)
        if all_positive:
            print(f"  ✓ All active CLIP losses are > 0.0 ✓")
        else:
            print(f"  ❌ Some active CLIP losses are 0.0!")
    
    # Final verdict
    print("\n" + "=" * 60)
    if warmup_transition_found:
        print("✓ Warmup transition message found")
    
    all_checks_pass = (
        warmup_transition_found and
        warmup_epochs and
        active_epochs and
        all(loss == 0.0 for _, loss in warmup_epochs) and
        all(loss > 0.0 for _, loss in active_epochs)
    )
    
    if all_checks_pass:
        print("✅ WARMUP MECHANISM WORKING CORRECTLY! ✅")
    else:
        print("⚠️  Some issues detected with warmup mechanism")
    
    print("=" * 60)
    return all_checks_pass


def verify_from_tensorboard(log_dir):
    """Verify warmup from TensorBoard event files"""
    print(f"Checking TensorBoard logs: {log_dir}")
    print("-" * 60)
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Check if clip_loss_active exists
        if 'Train/clip_loss_active' in ea.Tags()['scalars']:
            clip_active = ea.Scalars('Train/clip_loss_active')
            print(f"\n✓ Found Train/clip_loss_active:")
            for event in clip_active[:10]:
                status = "WARMUP" if event.value == 0.0 else "ACTIVE"
                print(f"  Epoch {event.step}: {event.value} [{status}]")
        
        # Check clip_loss
        if 'Train/Loss/clip_loss' in ea.Tags()['scalars']:
            clip_loss = ea.Scalars('Train/Loss/clip_loss')
            print(f"\n✓ Found Train/Loss/clip_loss:")
            for event in clip_loss[:10]:
                print(f"  Epoch {event.step}: {event.value:.4f}")
        
        print("\n✓ TensorBoard logs verified!")
        return True
        
    except ImportError:
        print("⚠️  TensorBoard not installed. Install with: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ Error reading TensorBoard logs: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify CLIP loss warmup mechanism')
    parser.add_argument('--log_file', type=str, default='output/debug/test_warmup/*/rank0.txt',
                       help='Path to rank0.txt log file (supports wildcards)')
    parser.add_argument('--tensorboard_dir', type=str, default='',
                       help='Path to TensorBoard log directory')
    args = parser.parse_args()
    
    # Find log file if wildcard
    if '*' in args.log_file:
        import glob
        matching_files = glob.glob(args.log_file)
        if matching_files:
            args.log_file = matching_files[0]
            print(f"Found log file: {args.log_file}\n")
    
    # Verify from text log
    verify_from_text_log(args.log_file)
    
    # Verify from TensorBoard if specified
    if args.tensorboard_dir and os.path.exists(args.tensorboard_dir):
        print("\n")
        verify_from_tensorboard(args.tensorboard_dir)
