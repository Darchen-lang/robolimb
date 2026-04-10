#include <Servo.h>
#include <math.h>

// ── Link lengths (mm) ─────────────────────────────────────────────────────
#define L1      200.00f
#define L2      195.00f
#define DX       0.0f
#define DZ      100.00f


// ── Servo gap between joints ──────────────────────────────────────────────
#define SERVO_GAP_MS    200

// ── Gripper angles ────────────────────────────────────────────────────────
#define GRIPPER_OPEN    120
#define GRIPPER_CLOSED   0

// ── Servo config ──────────────────────────────────────────────────────────
// reversed = true  → flips physical rotation direction
// reversed = false → normal direction

struct ServoConfig {
    float min_a, max_a;
    int   pin;
    int   min_us, max_us;
    bool  reversed;
};

//                   min°   max°  pin  minµs  maxµs   reversed
ServoConfig CONFIG[] = {
    {  0.0, 180.0,  9,  500, 2500, false },  // [0] Base     — 40kg
    {  0.0, 180.0,  5, 500, 2500, true },  // [1] Shoulder — set true to reverse
    {  0.0, 180.0, 11, 500, 2500, false },  // [2] Elbow    — set true to reverse
    {  0.0, 180.0, 12, 1000, 2000, false },  // [3] Gripper  — set true to reverse
};


// ── Per-servo motion profile (UPDATED: smaller steps + faster rate) ───────
struct MotionProfile { float stepDeg; int delayMs; };

MotionProfile PROFILE[] = {
    { 0.5f,  40 },   // [0] Base     — finer control, faster updates
    { 0.5f,  50 },   // [1] Shoulder — finer control, smooth ramp
    { 1.0f,  30 },   // [2] Elbow    — 1.0° per step, quick updates
    { 1.0f,  30 },   // [3] Gripper  — 1.0° per step, quick updates
};

Servo servos[4];
// float currentAngles[4] = {90, 90, 90, GRIPPER_OPEN};
float currentAngles[4] = {90, 166, 180, GRIPPER_OPEN};

const char* NAMES[] = {"Base", "Shoulder", "Elbow", "Gripper"};

// ── Move orders ───────────────────────────────────────────────────────────
const int MOVE_ORDER[] = {0, 2, 1};   // To target:  Base → Elbow → Shoulder
const int HOME_ORDER[] = {0, 1, 2};   // To home:    Base → Shoulder → Elbow

// ── Easing function: cubic ease-in-out ────────────────────────────────────
// t: normalized time from 0 to 1
// Returns: eased value from 0 to 1 with smooth acceleration/deceleration
float easeInOutCubic(float t) {
    if (t < 0.5f) {
        return 4.0f * t * t * t;  // ease in: slow start
    } else {
        float f = 2.0f * t - 2.0f;
        return 0.5f * f * f * f + 1.0f;  // ease out: slow end
    }
}

// ── Inverse Kinematics ────────────────────────────────────────────────────
bool ikArm(float x, float y, float z, float out[3]) {
    float theta_base = atan2f(y, x);

    float r_eff = sqrtf(x*x + y*y) - DX;
    float z_eff = z - DZ;

    float D = sqrtf(r_eff*r_eff + z_eff*z_eff);
    if (D > (L1 + L2) || D < fabsf(L1 - L2)) return false;

    float cos_el      = (L1*L1 + L2*L2 - D*D) / (2.0f * L1 * L2);
    float theta_elbow = M_PI - acosf(constrain(cos_el, -1.0f, 1.0f));

    float phi            = atan2f(z_eff, r_eff);
    float cos_psi        = (L1*L1 + D*D - L2*L2) / (2.0f * L1 * D);
    float theta_shoulder = phi + acosf(constrain(cos_psi, -1.0f, 1.0f));

    out[0] = theta_base     * 180.0f / M_PI + 90.0f;
    out[1] = theta_shoulder * 180.0f / M_PI + 90.0f;
    out[2] = theta_elbow    * 180.0f / M_PI + 90.0f;
    

    return true;
}

// ── Write angle to servo (handles reversal + pulse width mapping) ─────────
// This is the ONLY place physical angle is written to hardware.
// Reversal happens here — IK and motion logic never need to change.
void writeAngle(int index, float angleDeg) {
    // flip direction if reversed flag is set
    float physicalAngle = CONFIG[index].reversed
                          ? 180.0f - angleDeg
                          : angleDeg;

    // map angle to microseconds and write
    int us = map((int)physicalAngle, 0, 180,
                 CONFIG[index].min_us,
                 CONFIG[index].max_us);

    servos[index].writeMicroseconds(us);
}

// ── Move a single servo smoothly using easing curve ───────────────────────
// UPDATED: Uses cubic ease-in-out AND respects per-servo motion profiles
void moveSingleServo(int index, float target) {
    target = constrain(target, CONFIG[index].min_a, CONFIG[index].max_a);

    float startAngle = currentAngles[index];
    float diff = target - startAngle;

    // If already at target, return immediately
    if (fabsf(diff) < 0.5f) {
        currentAngles[index] = target;
        writeAngle(index, currentAngles[index]);
        return;
    }

    // Calculate steps based on stepDeg (respects original motion profile)
    float stepDeg = PROFILE[index].stepDeg;
    int numSteps = max(2, (int)(fabsf(diff) / stepDeg));  // at least 2 steps
    int delayMs = PROFILE[index].delayMs;

    for (int step = 0; step <= numSteps; step++) {
        float t = (float)step / numSteps;  // 0 to 1
        float easedT = easeInOutCubic(t);  // apply easing curve
        
        currentAngles[index] = startAngle + diff * easedT;
        writeAngle(index, currentAngles[index]);
        
        delay(delayMs);
    }

    // Ensure we end exactly at target
    currentAngles[index] = target;
    writeAngle(index, currentAngles[index]);
}

// ── Move arm joints in given order ───────────────────────────────────────
void writeInOrder(float targets[3], const int order[3]) {
    for (int i = 0; i < 3; i++) {
        int idx = order[i];
        Serial.print("Moving "); Serial.print(NAMES[idx]);
        Serial.print(" → ");
        Serial.print((int)constrain(targets[idx], 0, 180));
        Serial.println("°");

        moveSingleServo(idx, targets[idx]);

        Serial.print(NAMES[idx]); Serial.println(" done");
        delay(SERVO_GAP_MS);
    }
}

// ── Move gripper ──────────────────────────────────────────────────────────
void moveGripper(float angle) {
    Serial.print("Moving Gripper → ");
    Serial.print((int)angle); Serial.println("°");
    moveSingleServo(3, angle);
    Serial.println("Gripper done");
}


// ── Return to home ────────────────────────────────────────────────────────
void goHome() {
    float home[3] = {90, 175, 180};
    Serial.println("Returning to home...");
    writeInOrder(home, HOME_ORDER);
    delay(SERVO_GAP_MS);
    moveGripper(GRIPPER_OPEN);
    Serial.println("Home");
}


// ── Move arm to XYZ target ────────────────────────────────────────────────
bool moveTo(float x, float y, float z) {
    float angles[3];

    if (!ikArm(x, y, z, angles)) {
        Serial.print("[!] OUT OF REACH: ");
        Serial.print(x); Serial.print(", ");
        Serial.print(y); Serial.print(", ");
        Serial.println(z);
        return false;
    }

    Serial.println("─────────────────────");
    Serial.print("Target  X:"); Serial.print(x);
    Serial.print("  Y:");       Serial.print(y);
    Serial.print("  Z:");       Serial.println(z);
    Serial.print("Base     : "); Serial.println((int)constrain(angles[0], 0, 180));
    Serial.print("Shoulder : "); Serial.println((int)constrain(angles[1], 0, 180));
    Serial.print("Elbow    : "); Serial.println((int)constrain(angles[2], 0, 180));
    Serial.print("Gripper  : "); Serial.println(GRIPPER_CLOSED);

    writeInOrder(angles, MOVE_ORDER);
    delay(SERVO_GAP_MS);
    moveGripper(GRIPPER_CLOSED);

    Serial.println("OK");
    return true;
}

// ── Setup ─────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // Step 1: Hold all pins LOW to prevent floating signals
    for (int i = 0; i < 4; i++) {
        pinMode(CONFIG[i].pin, OUTPUT);
        digitalWrite(CONFIG[i].pin, LOW);
    }
    delay(500); // Wait for power rail to fully stabilize

    // Step 2: Attach and write home one by one with generous delay
    for (int i = 0; i < 4; i++) {
        servos[i].attach(CONFIG[i].pin, CONFIG[i].min_us, CONFIG[i].max_us);
        writeAngle(i, currentAngles[i]);
        delay(500); // stagger to avoid current spike
    }

    delay(1000); // Final settle before goHome()
    goHome();
}

// ── Loop — receive "x,y,z" over Serial ───────────────────────────────────
void loop() {
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        int c1 = line.indexOf(',');
        int c2 = line.indexOf(',', c1 + 1);

        if (c1 < 0 || c2 < 0) {
            Serial.println("[!] Bad format. Use: x,y,z");
            return;
        }

        float x = line.substring(0,      c1).toFloat();
        float y = line.substring(c1 + 1, c2).toFloat();
        float z = line.substring(c2 + 1).toFloat();

        bool ok = moveTo(x, y, z);
        Serial.println(ok ? "OK" : "FAIL");

        delay(2000);
        goHome();
    }
}
