/*
 * Copyright (c) 2026 L2jMobius
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.commons.time;

/**
 * Global time acceleration holder shared by the commons and gameserver modules.<br>
 * <ul>
 * <li>{@link #get()} returns the configured acceleration multiplier.</li>
 * <li>{@link #scaleDelay(long)} divides a real-time delay by the multiplier so that
 * scheduled tasks run that many times faster.</li>
 * </ul>
 * The multiplier is configured through the {@code TimeScale} property in {@code General.ini}.
 * @author rimmed
 */
public final class TimeScale
{
	private static volatile double SCALE = 1.0;
	
	private TimeScale()
	{
	}
	
	/**
	 * Sets the global time scale.
	 * @param scale the multiplier (values <= 0, NaN or infinity fall back to 1.0).
	 */
	public static synchronized void set(double scale)
	{
		if ((scale <= 0) || Double.isNaN(scale) || Double.isInfinite(scale))
		{
			scale = 1.0;
		}
		SCALE = scale;
	}
	
	/**
	 * @return the current time scale multiplier.
	 */
	public static double get()
	{
		return SCALE;
	}
	
	/**
	 * Scales a real-time delay so that it elapses {@code SCALE} times faster.<br>
	 * Handles zero/negative delays (returned unchanged) and avoids integer underflow
	 * by clamping the result to {@code 0}, and overflow by relying on double division.
	 * @param delay the delay in milliseconds.
	 * @return the scaled delay in milliseconds.
	 */
	public static long scaleDelay(long delay)
	{
		if ((delay <= 0) || (SCALE == 1.0))
		{
			return delay;
		}
		return Math.max(0L, Math.round(delay / SCALE));
	}
}
