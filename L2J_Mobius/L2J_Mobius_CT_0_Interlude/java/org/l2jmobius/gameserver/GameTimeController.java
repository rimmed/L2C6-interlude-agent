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
package org.l2jmobius.gameserver;

import org.l2jmobius.commons.time.TimeScale;

/**
 * Virtual game clock.<br>
 * The clock advances {@code TimeScale} times faster than real wall-clock time and is used
 * by game logic that stores absolute expiry timestamps (skill/item cooldowns, effect
 * durations) so those systems tick faster too.
 * @author rimmed
 */
public final class GameTimeController
{
	private static volatile long EPOCH_REAL = System.currentTimeMillis();
	private static volatile long EPOCH_GAME = EPOCH_REAL;
	
	private GameTimeController()
	{
	}
	
	/**
	 * Changes the time scale while keeping the virtual clock continuous.
	 * @param scale the new multiplier.
	 */
	public static synchronized void setTimeScale(double scale)
	{
		final long nowReal = System.currentTimeMillis();
		EPOCH_GAME = EPOCH_GAME + (long) ((nowReal - EPOCH_REAL) * TimeScale.get());
		EPOCH_REAL = nowReal;
		TimeScale.set(scale);
	}
	
	/**
	 * @return the configured time scale.
	 */
	public static double getTimeScale()
	{
		return TimeScale.get();
	}
	
	/**
	 * @return the current virtual game time in milliseconds.
	 */
	public static long currentTimeMillis()
	{
		return EPOCH_GAME + (long) ((System.currentTimeMillis() - EPOCH_REAL) * TimeScale.get());
	}
}
