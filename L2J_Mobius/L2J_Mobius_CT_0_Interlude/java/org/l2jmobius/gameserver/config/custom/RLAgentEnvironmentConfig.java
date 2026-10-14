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
package org.l2jmobius.gameserver.config.custom;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import org.l2jmobius.commons.util.ConfigReader;
import org.l2jmobius.gameserver.model.Location;
import org.l2jmobius.gameserver.model.item.holders.ItemHolder;
import org.l2jmobius.gameserver.model.item.type.CrystalType;

/**
 * This class loads all the RL agent environment related configurations.
 */
public class RLAgentEnvironmentConfig
{
	private static final Logger LOGGER = Logger.getLogger(RLAgentEnvironmentConfig.class.getName());
	
	// File
	private static final String RL_AGENT_ENVIRONMENT_CONFIG_FILE = "./config/Custom/RLAgentEnvironment.ini";
	
	/**
	 * Farm zone of a level bracket.
	 * @param minLevel first level of the bracket
	 * @param maxLevel last level of the bracket
	 * @param anchors teleport anchors, empty when {@code useStartPoint} is set
	 * @param useStartPoint {@code true} to use the race's newbie start point
	 */
	public record FarmZone(int minLevel, int maxLevel, List<Location> anchors, boolean useStartPoint)
	{
	}
	
	// Constants
	public static boolean RL_AGENT_ENABLED;
	public static Pattern RL_AGENT_ACCOUNT_PATTERN;
	public static boolean RL_APPLY_TO_EXISTING_CHARACTERS;
	public static boolean RL_AUTO_CREATE_CHARACTER;
	public static List<String> RL_NAME_PREFIXES;
	public static int RL_NAME_SUFFIX_LENGTH;
	public static int RL_MIN_LEVEL;
	public static int RL_MAX_LEVEL;
	public static boolean RL_AUTO_LEARN_SKILLS;
	public static boolean RL_AUTO_CLASS_TRANSFER;
	public static boolean RL_AUTO_EQUIP;
	/** Level at which each grade starts, ordered by level. */
	public static Map<Integer, CrystalType> RL_GRADE_LEVELS;
	public static int RL_SOULSHOT_COUNT;
	public static int RL_SPIRITSHOT_COUNT;
	public static boolean RL_USE_BLESSED_SPIRITSHOTS;
	public static boolean RL_AUTO_ENABLE_SHOTS;
	public static List<ItemHolder> RL_CONSUMABLES;
	public static boolean RL_TELEPORT_TO_FARM_ZONE;
	public static boolean RL_TELEPORT_ON_LOGIN;
	public static List<FarmZone> RL_FARM_ZONES;
	public static int RL_FARM_ZONE_SPREAD;
	/** Pick farm spots from the monster spawns ({@code auto}) instead of the fixed {@link #RL_FARM_ZONES} anchors. */
	public static boolean RL_FARM_AUTO;
	public static int RL_FARM_LEVEL_BELOW;
	public static int RL_FARM_LEVEL_ABOVE;
	/** Percent of an area's monsters that must lie in the level window. */
	public static int RL_FARM_MIN_SHARE;
	public static int RL_FARM_MIN_MONSTERS;
	public static int RL_FARM_MOBS_PER_AGENT;
	public static List<String> RL_FARM_EXCLUDED_SPAWN_FOLDERS;
	public static boolean RL_RESPAWN_ON_DEATH;
	public static int RL_RESPAWN_DELAY;
	
	public static void load()
	{
		final ConfigReader config = new ConfigReader(RL_AGENT_ENVIRONMENT_CONFIG_FILE);
		RL_AGENT_ENABLED = config.getBoolean("EnableRLAgentAutoSetup", false);
		RL_AGENT_ACCOUNT_PATTERN = Pattern.compile(config.getString("RLAgentAccountPattern", ".*"));
		RL_APPLY_TO_EXISTING_CHARACTERS = config.getBoolean("RLApplyToExistingCharacters", false);
		RL_AUTO_CREATE_CHARACTER = config.getBoolean("RLAutoCreateCharacter", true);
		RL_NAME_PREFIXES = new ArrayList<>();
		for (String prefix : config.getString("RLNamePrefixes", "Bot,Agent,RL").split(","))
		{
			final String clean = prefix.replaceAll("[^A-Za-z0-9]", "");
			if (!clean.isEmpty())
			{
				RL_NAME_PREFIXES.add(clean);
			}
		}
		if (RL_NAME_PREFIXES.isEmpty())
		{
			RL_NAME_PREFIXES.add("Bot");
		}
		RL_NAME_SUFFIX_LENGTH = Math.max(3, config.getInt("RLNameSuffixLength", 6));
		RL_MIN_LEVEL = Math.max(1, config.getInt("RLMinLevel", 1));
		RL_MAX_LEVEL = Math.max(RL_MIN_LEVEL, config.getInt("RLMaxLevel", 52));
		RL_AUTO_LEARN_SKILLS = config.getBoolean("RLAutoLearnSkills", true);
		RL_AUTO_CLASS_TRANSFER = config.getBoolean("RLAutoClassTransfer", true);
		RL_AUTO_EQUIP = config.getBoolean("RLAutoEquip", true);
		RL_GRADE_LEVELS = parseGradeLevels(config.getString("RLGradeLevels", "D:20;C:40;B:52;A:61;S:76"));
		RL_SOULSHOT_COUNT = config.getInt("RLSoulshotCount", 2000);
		RL_SPIRITSHOT_COUNT = config.getInt("RLSpiritshotCount", 1000);
		RL_USE_BLESSED_SPIRITSHOTS = config.getBoolean("RLUseBlessedSpiritshots", true);
		RL_AUTO_ENABLE_SHOTS = config.getBoolean("RLAutoEnableShots", true);
		RL_CONSUMABLES = parseItems(config.getString("RLConsumables", "1061:50;1539:50"));
		RL_TELEPORT_TO_FARM_ZONE = config.getBoolean("RLTeleportToFarmZone", true);
		RL_TELEPORT_ON_LOGIN = config.getBoolean("RLTeleportOnLogin", true);
		RL_FARM_ZONES = parseFarmZones(config.getString("RLFarmZones", "1-80:start"));
		RL_FARM_ZONE_SPREAD = Math.max(0, config.getInt("RLFarmZoneSpread", 150));
		RL_FARM_AUTO = config.getString("RLFarmZoneMode", "auto").trim().equalsIgnoreCase("auto");
		RL_FARM_LEVEL_BELOW = Math.max(0, config.getInt("RLFarmLevelBelow", 2));
		RL_FARM_LEVEL_ABOVE = Math.max(0, config.getInt("RLFarmLevelAbove", 1));
		RL_FARM_MIN_SHARE = Math.min(100, Math.max(0, config.getInt("RLFarmMinShare", 70)));
		RL_FARM_MIN_MONSTERS = Math.max(1, config.getInt("RLFarmMinMonsters", 4));
		RL_FARM_MOBS_PER_AGENT = Math.max(1, config.getInt("RLFarmMobsPerAgent", 5));
		RL_FARM_EXCLUDED_SPAWN_FOLDERS = new ArrayList<>();
		for (String folder : config.getString("RLFarmExcludedSpawnFolders", "Castles,Catacombs,SevenSigns").split(","))
		{
			if (!folder.isBlank())
			{
				RL_FARM_EXCLUDED_SPAWN_FOLDERS.add(folder.trim());
			}
		}
		RL_RESPAWN_ON_DEATH = config.getBoolean("RLRespawnOnDeath", true);
		RL_RESPAWN_DELAY = Math.max(0, config.getInt("RLRespawnDelay", 4000));
	}
	
	private static Map<Integer, CrystalType> parseGradeLevels(String value)
	{
		final Map<Integer, CrystalType> result = new TreeMap<>();
		for (String entry : value.split(";"))
		{
			final String[] parts = entry.trim().split(":");
			if (parts.length != 2)
			{
				continue;
			}
			try
			{
				result.put(Integer.parseInt(parts[1].trim()), CrystalType.valueOf(parts[0].trim().toUpperCase()));
			}
			catch (IllegalArgumentException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLGradeLevels entry: " + entry);
			}
		}
		return Collections.unmodifiableMap(result);
	}
	
	private static List<ItemHolder> parseItems(String value)
	{
		final List<ItemHolder> result = new ArrayList<>();
		for (String entry : value.split(";"))
		{
			final String[] parts = entry.trim().split(":");
			if (parts.length != 2)
			{
				continue;
			}
			try
			{
				result.add(new ItemHolder(Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim())));
			}
			catch (NumberFormatException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLConsumables entry: " + entry);
			}
		}
		return Collections.unmodifiableList(result);
	}
	
	private static List<FarmZone> parseFarmZones(String value)
	{
		final List<FarmZone> result = new ArrayList<>();
		for (String entry : value.split(";"))
		{
			final int colon = entry.indexOf(':');
			if (colon < 0)
			{
				continue;
			}
			try
			{
				final String[] range = entry.substring(0, colon).trim().split("-");
				final int minLevel = Integer.parseInt(range[0].trim());
				final int maxLevel = Integer.parseInt(range[range.length - 1].trim());
				final String anchorsValue = entry.substring(colon + 1).trim();
				if (anchorsValue.equalsIgnoreCase("start"))
				{
					result.add(new FarmZone(minLevel, maxLevel, List.of(), true));
					continue;
				}
				
				final List<Location> anchors = new ArrayList<>();
				for (String anchor : anchorsValue.split("\\|"))
				{
					final String[] xyz = anchor.trim().split(",");
					anchors.add(new Location(Integer.parseInt(xyz[0].trim()), Integer.parseInt(xyz[1].trim()), Integer.parseInt(xyz[2].trim())));
				}
				result.add(new FarmZone(minLevel, maxLevel, List.copyOf(anchors), false));
			}
			catch (RuntimeException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLFarmZones entry: " + entry);
			}
		}
		return Collections.unmodifiableList(result);
	}
}
