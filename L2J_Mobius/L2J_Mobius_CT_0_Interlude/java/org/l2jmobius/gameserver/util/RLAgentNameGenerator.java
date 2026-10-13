/*
 * Copyright (c) 2013 L2jMobius
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
package org.l2jmobius.gameserver.util;

import java.util.List;

import org.l2jmobius.commons.util.Rnd;
import org.l2jmobius.commons.util.StringUtil;
import org.l2jmobius.gameserver.config.PlayerConfig;
import org.l2jmobius.gameserver.config.ServerConfig;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.data.sql.CharInfoTable;
import org.l2jmobius.gameserver.data.xml.FakePlayerData;

/**
 * Generates unique character names for RL agent characters: a configured prefix ({@code Bot}, {@code Agent}, {@code RL}, ...) followed by a random alphanumeric suffix.<br>
 * Every candidate passes the same checks as a name typed in the character creation screen (length, alphanumeric, {@code CnameTemplate}, forbidden names, fake players) and must not exist yet.
 */
public class RLAgentNameGenerator
{
	private static final String SUFFIX_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789";
	private static final int MAX_NAME_LENGTH = 16;
	private static final int MAX_ATTEMPTS = 200;
	
	private RLAgentNameGenerator()
	{
	}
	
	/**
	 * Must be called while holding the {@link CharInfoTable} lock when the name is used for a new character, so no other creation can take it in between.
	 * @return a free, valid character name, or {@code null} if none was found
	 */
	public static String generate()
	{
		final List<String> prefixes = RLAgentEnvironmentConfig.RL_NAME_PREFIXES;
		final int suffixLength = Math.min(RLAgentEnvironmentConfig.RL_NAME_SUFFIX_LENGTH, MAX_NAME_LENGTH - 1);
		for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
		{
			String prefix = prefixes.get(Rnd.get(prefixes.size()));
			if ((prefix.length() + suffixLength) > MAX_NAME_LENGTH)
			{
				prefix = prefix.substring(0, MAX_NAME_LENGTH - suffixLength);
			}
			
			final StringBuilder name = new StringBuilder(prefix);
			for (int i = 0; i < suffixLength; i++)
			{
				name.append(SUFFIX_CHARACTERS.charAt(Rnd.get(SUFFIX_CHARACTERS.length())));
			}
			
			final String candidate = name.toString();
			if (isValid(candidate) && !CharInfoTable.getInstance().doesCharNameExist(candidate))
			{
				return candidate;
			}
		}
		return null;
	}
	
	private static boolean isValid(String name)
	{
		if (!StringUtil.isAlphaNumeric(name) || !ServerConfig.CHARNAME_TEMPLATE_PATTERN.matcher(name).matches())
		{
			return false;
		}
		
		for (String forbidden : PlayerConfig.FORBIDDEN_NAMES)
		{
			if (!forbidden.isEmpty() && name.toLowerCase().contains(forbidden.toLowerCase()))
			{
				return false;
			}
		}
		
		return FakePlayerData.getInstance().getProperName(name) == null;
	}
}
